from sympy.core.function import (Derivative, Function, Subs, diff)
from sympy.core.numbers import (E, I, Rational, pi)
from sympy.core.relational import Eq
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, symbols)
from sympy.functions.elementary.complexes import (im, re)
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.hyperbolic import acosh
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (atan2, cos, sin, tan)
from sympy.integrals.integrals import Integral
from sympy.polys.polytools import Poly
from sympy.series.order import O
from sympy.simplify.radsimp import collect
from sympy.solvers.ode import (classify_ode,
from sympy.solvers.ode.subscheck import checkodesol
from sympy.solvers.ode.ode import (classify_sysode,
from sympy.solvers.ode.nonhomogeneous import _undetermined_coefficients_match
from sympy.solvers.ode.single import LinearCoefficients
from sympy.solvers.deutils import ode_order
from sympy.testing.pytest import XFAIL, raises, slow
from sympy.utilities.misc import filldedent
@slow
def test_dsolve_options():
    eq = x * f(x).diff(x) + f(x)
    a = dsolve(eq, hint='all')
    b = dsolve(eq, hint='all', simplify=False)
    c = dsolve(eq, hint='all_Integral')
    keys = ['1st_exact', '1st_exact_Integral', '1st_homogeneous_coeff_best', '1st_homogeneous_coeff_subs_dep_div_indep', '1st_homogeneous_coeff_subs_dep_div_indep_Integral', '1st_homogeneous_coeff_subs_indep_div_dep', '1st_homogeneous_coeff_subs_indep_div_dep_Integral', '1st_linear', '1st_linear_Integral', 'Bernoulli', 'Bernoulli_Integral', 'almost_linear', 'almost_linear_Integral', 'best', 'best_hint', 'default', 'factorable', 'lie_group', 'nth_linear_euler_eq_homogeneous', 'order', 'separable', 'separable_Integral']
    Integral_keys = ['1st_exact_Integral', '1st_homogeneous_coeff_subs_dep_div_indep_Integral', '1st_homogeneous_coeff_subs_indep_div_dep_Integral', '1st_linear_Integral', 'Bernoulli_Integral', 'almost_linear_Integral', 'best', 'best_hint', 'default', 'factorable', 'nth_linear_euler_eq_homogeneous', 'order', 'separable_Integral']
    assert sorted(a.keys()) == keys
    assert a['order'] == ode_order(eq, f(x))
    assert a['best'] == Eq(f(x), C1 / x)
    assert dsolve(eq, hint='best') == Eq(f(x), C1 / x)
    assert a['default'] == 'factorable'
    assert a['best_hint'] == 'factorable'
    assert not a['1st_exact'].has(Integral)
    assert not a['separable'].has(Integral)
    assert not a['1st_homogeneous_coeff_best'].has(Integral)
    assert not a['1st_homogeneous_coeff_subs_dep_div_indep'].has(Integral)
    assert not a['1st_homogeneous_coeff_subs_indep_div_dep'].has(Integral)
    assert not a['1st_linear'].has(Integral)
    assert a['1st_linear_Integral'].has(Integral)
    assert a['1st_exact_Integral'].has(Integral)
    assert a['1st_homogeneous_coeff_subs_dep_div_indep_Integral'].has(Integral)
    assert a['1st_homogeneous_coeff_subs_indep_div_dep_Integral'].has(Integral)
    assert a['separable_Integral'].has(Integral)
    assert sorted(b.keys()) == keys
    assert b['order'] == ode_order(eq, f(x))
    assert b['best'] == Eq(f(x), C1 / x)
    assert dsolve(eq, hint='best', simplify=False) == Eq(f(x), C1 / x)
    assert b['default'] == 'factorable'
    assert b['best_hint'] == 'factorable'
    assert a['separable'] != b['separable']
    assert a['1st_homogeneous_coeff_subs_dep_div_indep'] != b['1st_homogeneous_coeff_subs_dep_div_indep']
    assert a['1st_homogeneous_coeff_subs_indep_div_dep'] != b['1st_homogeneous_coeff_subs_indep_div_dep']
    assert not b['1st_exact'].has(Integral)
    assert not b['separable'].has(Integral)
    assert not b['1st_homogeneous_coeff_best'].has(Integral)
    assert not b['1st_homogeneous_coeff_subs_dep_div_indep'].has(Integral)
    assert not b['1st_homogeneous_coeff_subs_indep_div_dep'].has(Integral)
    assert not b['1st_linear'].has(Integral)
    assert b['1st_linear_Integral'].has(Integral)
    assert b['1st_exact_Integral'].has(Integral)
    assert b['1st_homogeneous_coeff_subs_dep_div_indep_Integral'].has(Integral)
    assert b['1st_homogeneous_coeff_subs_indep_div_dep_Integral'].has(Integral)
    assert b['separable_Integral'].has(Integral)
    assert sorted(c.keys()) == Integral_keys
    raises(ValueError, lambda: dsolve(eq, hint='notarealhint'))
    raises(ValueError, lambda: dsolve(eq, hint='Liouville'))
    assert dsolve(f(x).diff(x) - 1 / f(x) ** 2, hint='all')['best'] == dsolve(f(x).diff(x) - 1 / f(x) ** 2, hint='best')
    assert dsolve(f(x) + f(x).diff(x) + sin(x).diff(x) + 1, f(x), hint='1st_linear_Integral') == Eq(f(x), (C1 + Integral((-sin(x).diff(x) - 1) * exp(Integral(1, x)), x)) * exp(-Integral(1, x)))