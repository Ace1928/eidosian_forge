import math
from enum import Enum
from functools import partial
from typing import List, Optional, Sequence, Tuple, Union
import torch
import torch._prims_common as utils
from torch import SymBool, SymFloat, Tensor
from torch._decomp import (
from torch._ops import OpOverload
from torch._prims import _prim_elementwise_meta, ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND
from torch._prims_common import (
from torch._prims_common.wrappers import (
from torch._refs import _broadcast_shapes, _maybe_broadcast
from torch.utils import _pytree as pytree
import torch._refs
import torch._refs.nn.functional
import torch._refs.special
def pool3d_shape_check(input: Tensor, nslices: int, kT: int, kH: int, kW: int, dT: int, dH: int, dW: int, pT: int, pH: int, pW: int, dilationT: int, dilationH: int, dilationW: int, itime: int, iheight: int, iwidth: int, otime: int, oheight: int, owidth: int, fn_name: str, check_input_size: bool=False):
    ndim = input.ndim
    torch._check(kT > 0 and kW > 0 and (kH > 0), lambda: f'kernel size should be greater than zero, but got kT: {kT}, kH: {kH}, kW: {kW}')
    torch._check(dT > 0 and dW > 0 and (dH > 0), lambda: f'stride should be greater than zero, but got dT: {dT}, dH: {dH}, dW: {dW}')
    torch._check(dilationT > 0 and dilationW > 0 and (dilationH > 0), lambda: f'dilation should be greater than zero, but got dilationT: {dilationT}, dilationH: {dilationH}, dilationW: {dilationW}')
    torch._check(ndim in (4, 5), lambda: f'{fn_name}: Expected 4D or 5D tensor for input, but got: {input.shape}')
    for i in range(ndim):
        if ndim == 5 and i == 0:
            continue
        torch._check(input.size(i) > 0, lambda: f"{fn_name}: Expected input's non-batch dimensions to have positive length, but input has a shape of {input.shape} and non-batch dimension {input.size(i)} has length zero!")
    if check_input_size:
        torch._check(itime >= kT and iheight >= kH and (iwidth >= kW), lambda: f'input image (T: {itime} H: {iheight} W: {iwidth}) smaller than kernel size (kT: {kT} kH: {kH} kW: {kW})')
    torch._check(kT / 2 >= pT and kW / 2 >= pW and (kH / 2 >= pH), lambda: f'pad should be smaller than or equal to half of kernel size, but got kT: {kT} kW: {kW} kH: {kH} padT: {pT} padW: {pW} padH: {pH}')
    torch._check(otime >= 1 and owidth >= 1 and (oheight >= 1), lambda: f'Given input size: ({nslices}x{itime}x{iheight}x{iwidth}). Calculated output size: ({nslices}x{otime}x{oheight}x{owidth}). Output size is too small')