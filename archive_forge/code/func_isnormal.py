from .ctx_base import StandardBaseContext
import math
import cmath
from . import math2
from . import function_docs
from .libmp import mpf_bernoulli, to_float, int_types
from . import libmp
def isnormal(ctx, x):
    if x:
        return x - x == 0.0
    return False