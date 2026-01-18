from functools import reduce, wraps
from itertools import repeat
from sympy.core import S, pi
from sympy.core.add import Add
from sympy.core.function import (
from sympy.core.mul import Mul
from sympy.core.numbers import igcd, ilcm
from sympy.core.sorting import default_sort_key
from sympy.core.symbol import Dummy
from sympy.core.traversal import postorder_traversal
from sympy.functions.combinatorial.factorials import factorial, rf
from sympy.functions.elementary.complexes import re, arg, Abs
from sympy.functions.elementary.exponential import exp, exp_polar
from sympy.functions.elementary.hyperbolic import cosh, coth, sinh, tanh
from sympy.functions.elementary.integers import ceiling
from sympy.functions.elementary.miscellaneous import Max, Min, sqrt
from sympy.functions.elementary.piecewise import piecewise_fold
from sympy.functions.elementary.trigonometric import cos, cot, sin, tan
from sympy.functions.special.bessel import besselj
from sympy.functions.special.delta_functions import Heaviside
from sympy.functions.special.gamma_functions import gamma
from sympy.functions.special.hyper import meijerg
from sympy.integrals import integrate, Integral
from sympy.integrals.meijerint import _dummy
from sympy.logic.boolalg import to_cnf, conjuncts, disjuncts, Or, And
from sympy.polys.polyroots import roots
from sympy.polys.polytools import factor, Poly
from sympy.polys.rootoftools import CRootOf
from sympy.utilities.iterables import iterable
from sympy.utilities.misc import debug
import sympy.integrals.laplace as _laplace
def process_conds(cond):
    """
        Turn ``cond`` into a strip (a, b), and auxiliary conditions.
        """
    from sympy.solvers.inequalities import _solve_inequality
    a = S.NegativeInfinity
    b = S.Infinity
    aux = S.true
    conds = conjuncts(to_cnf(cond))
    t = Dummy('t', real=True)
    for c in conds:
        a_ = S.Infinity
        b_ = S.NegativeInfinity
        aux_ = []
        for d in disjuncts(c):
            d_ = d.replace(re, lambda x: x.as_real_imag()[0]).subs(re(s), t)
            if not d.is_Relational or d.rel_op in ('==', '!=') or d_.has(s) or (not d_.has(t)):
                aux_ += [d]
                continue
            soln = _solve_inequality(d_, t)
            if not soln.is_Relational or soln.rel_op in ('==', '!='):
                aux_ += [d]
                continue
            if soln.lts == t:
                b_ = Max(soln.gts, b_)
            else:
                a_ = Min(soln.lts, a_)
        if a_ is not S.Infinity and a_ != b:
            a = Max(a_, a)
        elif b_ is not S.NegativeInfinity and b_ != a:
            b = Min(b_, b)
        else:
            aux = And(aux, Or(*aux_))
    return (a, b, aux)