from typing import Type
from sympy.core.add import Add
from sympy.core.basic import Basic
from sympy.core.containers import Tuple
from sympy.core.evalf import EvalfMixin
from sympy.core.expr import Expr
from sympy.core.function import expand
from sympy.core.logic import fuzzy_and
from sympy.core.mul import Mul
from sympy.core.power import Pow
from sympy.core.singleton import S
from sympy.core.symbol import Dummy, Symbol
from sympy.core.sympify import sympify, _sympify
from sympy.matrices import ImmutableMatrix, eye
from sympy.matrices.expressions import MatMul, MatAdd
from sympy.polys import Poly, rootof
from sympy.polys.polyroots import roots
from sympy.polys.polytools import (cancel, degree)
from sympy.series import limit
from mpmath.libmp.libmpf import prec_to_dps
def dc_gain(self):
    """
        Computes the gain of the response as the frequency approaches zero.

        The DC gain is infinite for systems with pure integrators.

        Examples
        ========

        >>> from sympy.abc import s, p, a, b
        >>> from sympy.physics.control.lti import TransferFunction
        >>> tf1 = TransferFunction(s + 3, s**2 - 9, s)
        >>> tf1.dc_gain()
        -1/3
        >>> tf2 = TransferFunction(p**2, p - 3 + p**3, p)
        >>> tf2.dc_gain()
        0
        >>> tf3 = TransferFunction(a*p**2 - b, s + b, s)
        >>> tf3.dc_gain()
        (a*p**2 - b)/b
        >>> tf4 = TransferFunction(1, s, s)
        >>> tf4.dc_gain()
        oo

        """
    m = Mul(self.num, Pow(self.den, -1, evaluate=False), evaluate=False)
    return limit(m, self.var, 0)