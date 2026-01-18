from ..activations import ACT2FN
from ..modeling_utils import PreTrainedModel
from ..utils import is_auto_awq_available, is_torch_available
from ..utils.quantization_config import AwqBackendPackingMethod, AwqConfig, AWQLinearVersion
def fuse_awq_modules(model, quantization_config):
    """
    Optionally fuse some modules in the model to speedup inference.

    Args:
        model (`~PreTrainedModel`):
            The model to fuse - note this model should have been converted into AWQ format beforehand.
        quantization_config (`Union[AwqConfig, dict]`):
            The quantization configuration to use.
    """
    if isinstance(quantization_config, dict):
        quantization_config = AwqConfig.from_dict(quantization_config)
    backend = quantization_config.backend
    modules_to_fuse = get_modules_to_fuse(model, quantization_config)
    modules_to_not_convert = getattr(quantization_config, 'modules_to_not_convert', None)
    if backend == AwqBackendPackingMethod.AUTOAWQ:
        from awq.modules.fused.attn import QuantAttentionFused
        from awq.modules.fused.mlp import QuantFusedMLP
        from awq.modules.fused.norm import FasterTransformerRMSNorm
    else:
        raise ValueError('Fusing is only supported for the AutoAWQ backend')
    for name, module in model.named_modules():
        if modules_to_not_convert is not None:
            if any((module_name_to_not_convert in name for module_name_to_not_convert in modules_to_not_convert)):
                continue
        _fuse_awq_layernorm(modules_to_fuse['layernorm'], module, FasterTransformerRMSNorm)
        _fuse_awq_mlp(model, name, modules_to_fuse['mlp'], module, QuantFusedMLP)
        _fuse_awq_attention_layers(model, module, modules_to_fuse, name, QuantAttentionFused)
    return model