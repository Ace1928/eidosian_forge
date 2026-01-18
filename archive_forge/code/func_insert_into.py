import itertools
import math
from typing import (
import numpy
from ..types import (
from ..util import get_array_module, is_xp_array, to_numpy
from .cblas import CBlas
def insert_into(self, shape, Xs):
    """Maybe don't need this? Just a quicky to get Jax working."""
    output = self.alloc(shape, dtype=Xs[0].dtype)
    for i, x in enumerate(Xs):
        output[i, :x.shape[0]] = x
    return output