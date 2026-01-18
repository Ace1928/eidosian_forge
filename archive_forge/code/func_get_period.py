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
def get_period(self):
    """
        Return a number $P$ such that $G(x*exp(I*P)) == G(x)$.

        Examples
        ========

        >>> from sympy import meijerg, pi, S
        >>> from sympy.abc import z

        >>> meijerg([1], [], [], [], z).get_period()
        2*pi
        >>> meijerg([pi], [], [], [], z).get_period()
        oo
        >>> meijerg([1, 2], [], [], [], z).get_period()
        oo
        >>> meijerg([1,1], [2], [1, S(1)/2, S(1)/3], [1], z).get_period()
        12*pi

        """

    def compute(l):
        for i, b in enumerate(l):
            if not b.is_Rational:
                return oo
            for j in range(i + 1, len(l)):
                if not Mod((b - l[j]).simplify(), 1):
                    return oo
        return reduce(ilcm, (x.q for x in l), 1)
    beta = compute(self.bm)
    alpha = compute(self.an)
    p, q = (len(self.ap), len(self.bq))
    if p == q:
        if oo in (alpha, beta):
            return oo
        return 2 * pi * ilcm(alpha, beta)
    elif p < q:
        return 2 * pi * beta
    else:
        return 2 * pi * alpha