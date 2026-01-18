import math
import torch
import torch.nn.functional as F
from torch.cuda.amp import custom_fwd, custom_bwd
import triton
import triton.language as tl
def layer_norm_fn(x, weight, bias, residual=None, x1=None, weight1=None, bias1=None, eps=1e-06, dropout_p=0.0, rowscale=None, prenorm=False, residual_in_fp32=False, is_rms_norm=False, return_dropout_mask=False):
    return LayerNormFn.apply(x, weight, bias, residual, x1, weight1, bias1, eps, dropout_p, rowscale, prenorm, residual_in_fp32, is_rms_norm, return_dropout_mask)