from ..sage_helper import _within_sage
import math
from ..snap import t3mlite as t3m
from ..snap.kernel_structures import *
from ..snap.mcomplex_base import *
from ..math_basics import correct_min
from .exceptions import *
def get_real_lengths(self):
    if not self._real_lengths_cache:
        self._real_lengths_cache = {side: abs(length) for side, length in self.lengths.items()}
    return self._real_lengths_cache