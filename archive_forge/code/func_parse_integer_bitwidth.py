import enum
import numpy as np
from .abstract import Dummy, Hashable, Literal, Number, Type
from functools import total_ordering, cached_property
from numba.core import utils
from numba.core.typeconv import Conversion
from numba.np import npdatetime_helpers
def parse_integer_bitwidth(name):
    for prefix in ('int', 'uint'):
        if name.startswith(prefix):
            bitwidth = int(name[len(prefix):])
    return bitwidth