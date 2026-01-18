from ..activations import ACT2FN
from ..modeling_utils import PreTrainedModel
from ..utils import is_auto_awq_available, is_torch_available
from ..utils.quantization_config import AwqBackendPackingMethod, AwqConfig, AWQLinearVersion
def get_modules_to_fuse(model, quantization_config):
    """
    Returns the fusing mapping given the quantization config and the model

    Args:
        model (`~PreTrainedModel`):
            The model to fuse - note this model should have been converted into AWQ format beforehand.
        quantization_config (`~transformers.quantization_config.AWQConfig`):
            The quantization configuration to use.
    """
    if not isinstance(model, PreTrainedModel):
        raise ValueError(f'The model should be an instance of `PreTrainedModel`, got {model.__class__.__name__}')
    if quantization_config.modules_to_fuse is not None:
        current_fused_mapping = quantization_config.modules_to_fuse
        current_fused_mapping['max_seq_len'] = quantization_config.fuse_max_seq_len
    elif model.config.model_type in AWQ_FUSED_MAPPINGS:
        current_fused_mapping = AWQ_FUSED_MAPPINGS[model.config.model_type]
        if not hasattr(model.config, 'text_config'):
            config = model.config
        else:
            config = model.config.text_config
        hidden_size = config.hidden_size
        num_attention_heads = config.num_attention_heads
        num_key_value_heads = getattr(config, 'num_key_value_heads', num_attention_heads)
        current_fused_mapping['hidden_size'] = hidden_size
        current_fused_mapping['num_attention_heads'] = num_attention_heads
        current_fused_mapping['num_key_value_heads'] = num_key_value_heads
        current_fused_mapping['max_seq_len'] = quantization_config.fuse_max_seq_len
    else:
        raise ValueError('Fusing mapping not found either on the quantization config or the supported `AWQ_FUSED_MAPPINGS`. Please pass a `fused_mapping` argument in the `quantization_config` or raise an issue on transformers https://github.com/huggingface/transformers to add its support.')
    return current_fused_mapping