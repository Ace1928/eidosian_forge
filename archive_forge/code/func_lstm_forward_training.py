import itertools
import math
from typing import (
import numpy
from ..types import (
from ..util import get_array_module, is_xp_array, to_numpy
from .cblas import CBlas
def lstm_forward_training(self, params: Floats1d, H0: Floats3d, C0: Floats3d, X: Floats2d, size_at_t: Ints1d) -> Tuple[Floats2d, Tuple]:
    assert H0.shape == C0.shape
    assert H0.shape[1] == C0.shape[1]
    Y, fwd_state = lstm_forward_training(params, H0, C0, X, size_at_t)
    return (Y, fwd_state)