import dropout_layer_norm
import torch
from torch.nn import init
def dropout_add_layer_norm_subset(x0, residual, weight, bias, dropout_p, epsilon, layerscale=None, x0_subset=None, out_subset=None, rowscale_const=1.0, out_numrows=0, prenorm=False, residual_in_fp32=False, return_dropout_mask=False):
    """residual_in_fp32 only has an effect if residual is None.
    Otherwise residual dtype is residual.dtype.
    """
    return DropoutAddLayerNormSubsetFn.apply(x0, residual, weight, bias, layerscale, x0_subset, out_subset, dropout_p, epsilon, rowscale_const, out_numrows, residual_in_fp32, prenorm, False, return_dropout_mask)