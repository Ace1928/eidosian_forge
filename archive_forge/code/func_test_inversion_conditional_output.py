from sympy.core.function import expand_func
from sympy.core.numbers import (I, Rational, oo, pi)
from sympy.core.singleton import S
from sympy.core.sorting import default_sort_key
from sympy.functions.elementary.complexes import Abs, arg, re, unpolarify
from sympy.functions.elementary.exponential import (exp, exp_polar, log)
from sympy.functions.elementary.hyperbolic import cosh, acosh
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.piecewise import Piecewise, piecewise_fold
from sympy.functions.elementary.trigonometric import (cos, sin, sinc, asin)
from sympy.functions.special.error_functions import (erf, erfc)
from sympy.functions.special.gamma_functions import (gamma, polygamma)
from sympy.functions.special.hyper import (hyper, meijerg)
from sympy.integrals.integrals import (Integral, integrate)
from sympy.simplify.hyperexpand import hyperexpand
from sympy.simplify.simplify import simplify
from sympy.integrals.meijerint import (_rewrite_single, _rewrite1,
from sympy.testing.pytest import slow
from sympy.core.random import (verify_numerically,
from sympy.abc import x, y, a, b, c, d, s, t, z
def test_inversion_conditional_output():
    from sympy.core.symbol import Symbol
    from sympy.integrals.transforms import InverseLaplaceTransform
    a = Symbol('a', positive=True)
    F = sqrt(pi / a) * exp(-2 * sqrt(a) * sqrt(s))
    f = meijerint_inversion(F, s, t)
    assert not f.is_Piecewise
    b = Symbol('b', real=True)
    F = F.subs(a, b)
    f2 = meijerint_inversion(F, s, t)
    assert f2.is_Piecewise
    assert f2.args[0][0] == f.subs(a, b)
    assert f2.args[-1][1]
    ILT = InverseLaplaceTransform(F, s, t, None)
    assert f2.args[-1][0] == ILT or f2.args[-1][0] == ILT.as_integral