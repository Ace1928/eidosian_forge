import itertools
import math
from typing import (
import numpy
from ..types import (
from ..util import get_array_module, is_xp_array, to_numpy
from .cblas import CBlas
def backprop_lstm(self, dY: Floats2d, lengths: Ints1d, params: Floats1d, fwd_state: Tuple) -> Tuple[Floats2d, Floats1d]:
    dX, d_params = backprop_lstm(dY, lengths, params, fwd_state)
    return (dX, d_params)