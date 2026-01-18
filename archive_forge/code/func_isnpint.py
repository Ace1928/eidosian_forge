from .ctx_base import StandardBaseContext
import math
import cmath
from . import math2
from . import function_docs
from .libmp import mpf_bernoulli, to_float, int_types
from . import libmp
def isnpint(ctx, x):
    if type(x) is complex:
        if x.imag:
            return False
        x = x.real
    return x <= 0.0 and round(x) == x