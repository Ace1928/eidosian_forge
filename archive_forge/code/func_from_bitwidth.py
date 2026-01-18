import enum
import numpy as np
from .abstract import Dummy, Hashable, Literal, Number, Type
from functools import total_ordering, cached_property
from numba.core import utils
from numba.core.typeconv import Conversion
from numba.np import npdatetime_helpers
@classmethod
def from_bitwidth(cls, bitwidth, signed=True):
    name = ('int%d' if signed else 'uint%d') % bitwidth
    return cls(name)