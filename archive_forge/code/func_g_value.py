from collections import OrderedDict
from functools import reduce
import math
from operator import add
from ..units import get_derived_unit, default_units, energy, concentration
from ..util._dimensionality import dimension_codes, base_registry
from ..util.pyutil import memoize, deprecated
from ..util._expr import Expr, UnaryWrapper, Symbol
@deprecated(use_instead='Radiolytic.all_args')
def g_value(self, variables, backend=math, **kwargs):
    g_val, = self.all_args(variables, backend=backend, **kwargs)
    return g_val