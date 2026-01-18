import torch
from apex._autocast_utils import _cast_if_autocast_enabled
from apex.transformer.enums import AttnMaskType
from fused_softmax_lib import (
def scaled_masked_softmax(inputs, mask, scale):
    args = _cast_if_autocast_enabled(inputs, mask, scale)
    with torch.cuda.amp.autocast(enabled=False):
        return ScaledMaskedSoftmax.apply(*args)