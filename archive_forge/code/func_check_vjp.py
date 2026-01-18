from functools import partial
from itertools import product
from .core import make_vjp, make_jvp, vspace
from .util import subvals
from .wrap_util import unary_to_nary, get_name
def check_vjp(f, x):
    vjp, y = make_vjp(f, x)
    jvp = make_numerical_jvp(f, x)
    x_vs, y_vs = (vspace(x), vspace(y))
    x_v, y_v = (x_vs.randn(), y_vs.randn())
    vjp_y = x_vs.covector(vjp(y_vs.covector(y_v)))
    assert vspace(vjp_y) == x_vs
    vjv_exact = x_vs.inner_prod(x_v, vjp_y)
    vjv_numeric = y_vs.inner_prod(y_v, jvp(x_v))
    assert scalar_close(vjv_numeric, vjv_exact), 'Derivative (VJP) check of {} failed with arg {}:\nanalytic: {}\nnumeric:  {}'.format(get_name(f), x, vjv_exact, vjv_numeric)