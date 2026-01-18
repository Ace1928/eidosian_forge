from functools import reduce
from sympy.core import S, ilcm, Mod
from sympy.core.add import Add
from sympy.core.expr import Expr
from sympy.core.function import Function, Derivative, ArgumentIndexError
from sympy.core.containers import Tuple
from sympy.core.mul import Mul
from sympy.core.numbers import I, pi, oo, zoo
from sympy.core.relational import Ne
from sympy.core.sorting import default_sort_key
from sympy.core.symbol import Dummy
from sympy.functions import (sqrt, exp, log, sin, cos, asin, atan,
from sympy.functions import factorial, RisingFactorial
from sympy.functions.elementary.complexes import Abs, re, unpolarify
from sympy.functions.elementary.exponential import exp_polar
from sympy.functions.elementary.integers import ceiling
from sympy.functions.elementary.piecewise import Piecewise
from sympy.logic.boolalg import (And, Or)
@property
def convergence_statement(self):
    """ Return a condition on z under which the series converges. """
    R = self.radius_of_convergence
    if R == 0:
        return False
    if R == oo:
        return True
    e = self.eta
    z = self.argument
    c1 = And(re(e) < 0, abs(z) <= 1)
    c2 = And(0 <= re(e), re(e) < 1, abs(z) <= 1, Ne(z, 1))
    c3 = And(re(e) >= 1, abs(z) < 1)
    return Or(c1, c2, c3)