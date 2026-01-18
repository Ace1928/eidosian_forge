import math
import torch
import torch.nn.functional as F
from torch.cuda.amp import custom_fwd, custom_bwd
import triton
import triton.language as tl
def layer_norm_linear_fn(x, norm_weight, norm_bias, linear_weight, linear_bias, residual=None, eps=1e-06, prenorm=False, residual_in_fp32=False, is_rms_norm=False):
    return LayerNormLinearFn.apply(x, norm_weight, norm_bias, linear_weight, linear_bias, residual, eps, prenorm, residual_in_fp32, is_rms_norm)