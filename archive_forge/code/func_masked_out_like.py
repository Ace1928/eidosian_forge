from functools import partial
import torch
from .binary import (
from .core import is_masked_tensor, MaskedTensor, _get_data, _masks_match, _maybe_get_mask
from .passthrough import (
from .reductions import (
from .unary import (
def masked_out_like(mt):
    return MaskedTensor(mt.get_data(), torch.zeros_like(mt.get_mask()).bool())