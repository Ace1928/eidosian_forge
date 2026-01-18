from sympy.core.random import randrange
from sympy.simplify.hyperexpand import (ShiftA, ShiftB, UnShiftA, UnShiftB,
from sympy.concrete.summations import Sum
from sympy.core.containers import Tuple
from sympy.core.expr import Expr
from sympy.core.numbers import I
from sympy.core.singleton import S
from sympy.core.symbol import symbols
from sympy.functions.combinatorial.factorials import binomial
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.special.hyper import (hyper, meijerg)
from sympy.abc import z, a, b, c
from sympy.testing.pytest import XFAIL, raises, slow, ON_CI, skip
from sympy.core.random import verify_numerically as tn
from sympy.core.numbers import (Rational, pi)
from sympy.functions.elementary.exponential import (exp, exp_polar, log)
from sympy.functions.elementary.hyperbolic import atanh
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (asin, cos, sin)
from sympy.functions.special.bessel import besseli
from sympy.functions.special.error_functions import erf
from sympy.functions.special.gamma_functions import (gamma, lowergamma)
def test_meijerg_formulae():
    from sympy.simplify.hyperexpand import MeijerFormulaCollection
    formulae = MeijerFormulaCollection().formulae
    for sig in formulae:
        for formula in formulae[sig]:
            g = meijerg(formula.func.an, formula.func.ap, formula.func.bm, formula.func.bq, formula.z)
            rep = {}
            for sym in formula.symbols:
                rep[sym] = randcplx()
            g = g.subs(rep)
            closed_form = formula.closed_form.subs(rep)
            z = formula.z
            assert tn(g, closed_form, z)
            cl = (formula.C * formula.B)[0].subs(rep)
            assert tn(closed_form, cl, z)
            deriv1 = z * formula.B.diff(z)
            deriv2 = formula.M * formula.B
            for d1, d2 in zip(deriv1, deriv2):
                assert tn(d1.subs(rep), d2.subs(rep), z)