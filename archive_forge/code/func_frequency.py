from sympy.core.basic import Basic
from sympy.core.expr import Expr
from sympy.core.function import Derivative, Function
from sympy.core.numbers import (Number, pi, I)
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, symbols)
from sympy.core.sympify import _sympify, sympify
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (atan2, cos, sin)
from sympy.physics.units import speed_of_light, meter, second
@property
def frequency(self):
    """
        Returns the frequency of the wave,
        in cycles per second.

        Examples
        ========

        >>> from sympy import symbols
        >>> from sympy.physics.optics import TWave
        >>> A, phi, f = symbols('A, phi, f')
        >>> w = TWave(A, f, phi)
        >>> w.frequency
        f
        """
    return self.args[1]