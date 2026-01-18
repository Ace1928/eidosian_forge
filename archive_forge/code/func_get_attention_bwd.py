from typing import Callable, Optional, Tuple, cast
from ..config import registry
from ..model import Model
from ..types import Floats2d, Ragged
from ..util import get_width
from .noop import noop
def get_attention_bwd(d_attention):
    d_attention = ops.backprop_softmax_sequences(d_attention, attention, lengths)
    dQ = ops.gemm(K, d_attention, trans1=True)
    dY = ops.xp.outer(d_attention, Q)
    dX = K_bp(dY)
    return (dQ, dX)