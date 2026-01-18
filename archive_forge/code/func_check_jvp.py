from functools import partial
from itertools import product
from .core import make_vjp, make_jvp, vspace
from .util import subvals
from .wrap_util import unary_to_nary, get_name
def check_jvp(f, x):
    jvp = make_jvp(f, x)
    jvp_numeric = make_numerical_jvp(f, x)
    x_v = vspace(x).randn()
    check_equivalent(jvp(x_v)[1], jvp_numeric(x_v))