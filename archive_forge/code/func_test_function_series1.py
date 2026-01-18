from sympy.core.function import Function
from sympy.core.sympify import sympify
from sympy.functions.elementary.hyperbolic import tanh
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.series.limits import limit
from sympy.abc import x
def test_function_series1():
    """Create our new "sin" function."""

    class my_function(Function):

        def fdiff(self, argindex=1):
            return cos(self.args[0])

        @classmethod
        def eval(cls, arg):
            arg = sympify(arg)
            if arg == 0:
                return sympify(0)
    assert my_function(x).series(x, 0, 10) == sin(x).series(x, 0, 10)
    assert limit(my_function(x) / x, x, 0) == 1