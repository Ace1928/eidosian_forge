from sympy.integrals.laplace import (
from sympy.core.function import Function, expand_mul
from sympy.core import EulerGamma, Subs, Derivative, diff
from sympy.core.exprtools import factor_terms
from sympy.core.numbers import I, oo, pi
from sympy.core.singleton import S
from sympy.core.symbol import Symbol, symbols
from sympy.simplify.simplify import simplify
from sympy.functions.elementary.complexes import Abs, re
from sympy.functions.elementary.exponential import exp, log, exp_polar
from sympy.functions.elementary.hyperbolic import cosh, sinh, coth, asinh
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import atan, cos, sin
from sympy.functions.special.gamma_functions import lowergamma, gamma
from sympy.functions.special.delta_functions import DiracDelta, Heaviside
from sympy.functions.special.zeta_functions import lerchphi
from sympy.functions.special.error_functions import (
from sympy.functions.special.bessel import besseli, besselj, besselk, bessely
from sympy.testing.pytest import slow, warns_deprecated_sympy
from sympy.matrices import Matrix, eye
from sympy.abc import s
def simp_hyp(expr):
    return factor_terms(expand_mul(expr)).rewrite(sin)