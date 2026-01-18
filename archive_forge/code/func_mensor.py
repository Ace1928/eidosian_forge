from sympy.core.numbers import Rational
from sympy.core.singleton import S
from sympy.core.relational import is_eq
from sympy.functions.elementary.complexes import (conjugate, im, re, sign)
from sympy.functions.elementary.exponential import (exp, log as ln)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (acos, asin, atan2)
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.simplify.trigsimp import trigsimp
from sympy.integrals.integrals import integrate
from sympy.matrices.dense import MutableDenseMatrix as Matrix
from sympy.core.sympify import sympify, _sympify
from sympy.core.expr import Expr
from sympy.core.logic import fuzzy_not, fuzzy_or
from mpmath.libmp.libmpf import prec_to_dps
def mensor(self):
    """
        Returns the natural logarithm of the norm(magnitude) of the quaternion.

        Examples
        ========

        >>> from sympy.algebras.quaternion import Quaternion
        >>> q = Quaternion(2, 4, 2, 4)
        >>> q.mensor()
        log(2*sqrt(10))
        >>> q.norm()
        2*sqrt(10)

        See Also
        ========

        norm

        """
    return ln(self.norm())