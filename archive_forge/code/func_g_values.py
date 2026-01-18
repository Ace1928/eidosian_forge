from collections import OrderedDict
from functools import reduce
import math
from operator import add
from ..units import get_derived_unit, default_units, energy, concentration
from ..util._dimensionality import dimension_codes, base_registry
from ..util.pyutil import memoize, deprecated
from ..util._expr import Expr, UnaryWrapper, Symbol
def g_values(self, *args, **kwargs):
    return OrderedDict(zip(self.parameter_keys[1:], self.all_args(*args, **kwargs)))