from functools import partial
from itertools import product
from .core import make_vjp, make_jvp, vspace
from .util import subvals
from .wrap_util import unary_to_nary, get_name
def scalar_close(a, b):
    return abs(a - b) < TOL or abs(a - b) / abs(a + b) < RTOL