import torch
from apex._autocast_utils import _cast_if_autocast_enabled
from apex.transformer.enums import AttnMaskType
from fused_softmax_lib import (
def scaled_upper_triang_masked_softmax(inputs, _, scale):
    b, np, sq, sk = inputs.size()
    assert sq == sk, 'causal mask is only for self attention'
    inputs = inputs.view(-1, sq, sk)
    args = _cast_if_autocast_enabled(inputs, scale)
    with torch.cuda.amp.autocast(enabled=False):
        probs = ScaledUpperTriangMaskedSoftmax.apply(*args)
    return probs.view(b, np, sq, sk)