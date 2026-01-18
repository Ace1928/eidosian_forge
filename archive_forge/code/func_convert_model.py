import torch.nn as nn
from .imports import is_fp8_available
def convert_model(model, to_transformer_engine=True, _convert_linear=True, _convert_ln=True):
    """
    Recursively converts the linear and layernorm layers of a model to their `transformers_engine` counterpart.
    """
    if not is_fp8_available():
        raise ImportError('Using `convert_model` requires transformer_engine to be installed.')
    for name, module in model.named_children():
        if isinstance(module, nn.Linear) and to_transformer_engine and _convert_linear:
            if any((p % 16 != 0 for p in module.weight.shape)):
                return
            has_bias = module.bias is not None
            te_module = te.Linear(module.in_features, module.out_features, bias=has_bias, params_dtype=module.weight.dtype)
            te_module.weight.copy_(module.weight)
            if has_bias:
                te_module.bias.copy_(module.bias)
            setattr(model, name, te_module)
        elif isinstance(module, nn.LayerNorm) and to_transformer_engine and _convert_ln:
            te_module = te.LayerNorm(module.normalized_shape[0], eps=module.eps, params_dtype=module.weight.dtype)
            te_module.weight.copy_(module.weight)
            te_module.bias.copy_(module.bias)
            setattr(model, name, te_module)
        elif isinstance(module, te.Linear) and (not to_transformer_engine) and _convert_linear:
            has_bias = module.bias is not None
            new_module = nn.Linear(module.in_features, module.out_features, bias=has_bias, params_dtype=module.weight.dtype)
            new_module.weight.copy_(module.weight)
            if has_bias:
                new_module.bias.copy_(module.bias)
            setattr(model, name, new_module)
        elif isinstance(module, te.LayerNorm) and (not to_transformer_engine) and _convert_ln:
            new_module = nn.LayerNorm(module.normalized_shape[0], eps=module.eps, params_dtype=module.weight.dtype)
            new_module.weight.copy_(module.weight)
            new_module.bias.copy_(module.bias)
            setattr(model, name, new_module)
        else:
            convert_model(module, to_transformer_engine=to_transformer_engine, _convert_linear=_convert_linear, _convert_ln=_convert_ln)