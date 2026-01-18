from sympy.core.function import Function
from sympy.core.sympify import sympify
from sympy.functions.elementary.hyperbolic import tanh
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.series.limits import limit
from sympy.abc import x
def test_function_series3():
    """
    Test our easy "tanh" function.

    This test tests two things:
      * that the Function interface works as expected and it's easy to use
      * that the general algorithm for the series expansion works even when the
        derivative is defined recursively in terms of the original function,
        since tanh(x).diff(x) == 1-tanh(x)**2
    """

    class mytanh(Function):

        def fdiff(self, argindex=1):
            return 1 - mytanh(self.args[0]) ** 2

        @classmethod
        def eval(cls, arg):
            arg = sympify(arg)
            if arg == 0:
                return sympify(0)
    e = tanh(x)
    f = mytanh(x)
    assert e.series(x, 0, 6) == f.series(x, 0, 6)