from sympy.core.function import Function
from sympy.core.sympify import sympify
from sympy.functions.elementary.hyperbolic import tanh
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.series.limits import limit
from sympy.abc import x
def test_function_series2():
    """Create our new "cos" function."""

    class my_function2(Function):

        def fdiff(self, argindex=1):
            return -sin(self.args[0])

        @classmethod
        def eval(cls, arg):
            arg = sympify(arg)
            if arg == 0:
                return sympify(1)
    assert my_function2(x).series(x, 0, 10) == cos(x).series(x, 0, 10)