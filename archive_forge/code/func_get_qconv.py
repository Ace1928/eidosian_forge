from typing import Optional, List, TypeVar
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.ao.nn.intrinsic as nni
import torch.ao.nn.intrinsic.qat as nniqat
from torch._ops import ops
from torch.nn.common_types import _size_1_t
from torch.nn.modules.utils import _single, _pair, _triple
from torch.nn.utils import fuse_conv_bn_weights
from .utils import _quantize_weight, WeightedQuantizedModule
@classmethod
def get_qconv(cls, mod, activation_post_process, weight_post_process=None):
    """Creates a qconv object and returns it.
        """
    if weight_post_process is None:
        weight_post_process = mod.qconfig.weight()
    weight_post_process(mod.weight)
    assert weight_post_process.dtype == torch.qint8, 'Weight observer must have a dtype of qint8'
    qweight = _quantize_weight(mod.weight.float(), weight_post_process)
    qconv = cls(mod.in_channels, mod.out_channels, mod.kernel_size, mod.stride, mod.padding, mod.dilation, mod.groups, mod.bias is not None, mod.padding_mode)
    qconv.set_weight_bias(qweight, mod.bias)
    if activation_post_process is None or activation_post_process.dtype == torch.float:
        return qconv
    else:
        act_scale, act_zp = activation_post_process.calculate_qparams()
        qconv.scale = float(act_scale)
        qconv.zero_point = int(act_zp)
        return qconv