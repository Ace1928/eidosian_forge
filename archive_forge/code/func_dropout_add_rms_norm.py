import torch
from torch.nn import init
from flash_attn.ops.layer_norm import (
def dropout_add_rms_norm(x0, residual, weight, bias, dropout_p, epsilon, rowscale=None, layerscale=None, prenorm=False, residual_in_fp32=False, return_dropout_mask=False):
    """residual_in_fp32 only has an effect if residual is None.
    Otherwise residual dtype is residual.dtype.
    """
    return DropoutAddLayerNormFn.apply(x0, residual, weight, bias, rowscale, layerscale, dropout_p, epsilon, residual_in_fp32, prenorm, True, return_dropout_mask)