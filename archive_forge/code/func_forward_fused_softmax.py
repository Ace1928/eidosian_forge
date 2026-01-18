import torch
from apex._autocast_utils import _cast_if_autocast_enabled
from apex.transformer.enums import AttnMaskType
from fused_softmax_lib import (
def forward_fused_softmax(self, input, mask):
    scale = self.scale if self.scale is not None else 1.0
    return self.fused_softmax_func(input, mask, scale)