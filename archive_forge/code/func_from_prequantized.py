import copy
from typing import Any, Dict, Optional, TypeVar, Union, overload
import warnings
import torch
from torch import Tensor, device, dtype, nn
import torch.nn.functional as F
import bitsandbytes as bnb
from bitsandbytes.autograd._functions import get_tile_inds, undo_layout
from bitsandbytes.functional import QuantState
from bitsandbytes.optim import GlobalOptimManager
from bitsandbytes.utils import OutlierTracer
@classmethod
def from_prequantized(cls, data: torch.Tensor, quantized_stats: Dict[str, Any], requires_grad: bool=False, device='cuda', **kwargs) -> 'Params4bit':
    self = torch.Tensor._make_subclass(cls, data.to(device))
    self.requires_grad = requires_grad
    self.quant_state = QuantState.from_dict(qs_dict=quantized_stats, device=device)
    self.blocksize = self.quant_state.blocksize
    self.compress_statistics = self.quant_state.nested
    self.quant_type = self.quant_state.quant_type
    self.bnb_quantized = True
    return self