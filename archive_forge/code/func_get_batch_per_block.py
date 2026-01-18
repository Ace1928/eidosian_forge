import torch
from apex._autocast_utils import _cast_if_autocast_enabled
from apex.transformer.enums import AttnMaskType
from fused_softmax_lib import (
@staticmethod
def get_batch_per_block(sq, sk, b, np):
    return scaled_masked_softmax_get_batch_per_block(sq, sk, b, np)