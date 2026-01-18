from sympy.concrete.summations import Sum
from sympy.functions.elementary.exponential import log
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.utilities.lambdify import lambdify
from sympy.abc import x, i, a, b
from sympy.codegen.numpy_nodes import logaddexp
from sympy.printing.numpy import CuPyPrinter, _cupy_known_constants, _cupy_known_functions
from sympy.testing.pytest import skip
from sympy.external import import_module
def test_cupy_sum():
    if not cp:
        skip('CuPy not installed')
    s = Sum(x ** i, (i, a, b))
    f = lambdify((a, b, x), s, 'cupy')
    a_, b_ = (0, 10)
    x_ = cp.linspace(-1, +1, 10)
    assert cp.allclose(f(a_, b_, x_), sum((x_ ** i_ for i_ in range(a_, b_ + 1))))
    s = Sum(i * x, (i, a, b))
    f = lambdify((a, b, x), s, 'numpy')
    a_, b_ = (0, 10)
    x_ = cp.linspace(-1, +1, 10)
    assert cp.allclose(f(a_, b_, x_), sum((i_ * x_ for i_ in range(a_, b_ + 1))))