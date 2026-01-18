from sympy.core.function import (Derivative, Function, diff)
from sympy.core.mul import Mul
from sympy.core.numbers import (I, Rational, pi)
from sympy.core.relational import Eq
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, symbols)
from sympy.functions.elementary.hyperbolic import sinh
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.matrices.dense import Matrix
from sympy.core.containers import Tuple
from sympy.functions import exp, cos, sin, log, Ci, Si, erf, erfi
from sympy.matrices import dotprodsimp, NonSquareMatrixError
from sympy.solvers.ode import dsolve
from sympy.solvers.ode.ode import constant_renumber
from sympy.solvers.ode.subscheck import checksysodesol
from sympy.solvers.ode.systems import (_classify_linear_system, linear_ode_to_matrix,
from sympy.functions import airyai, airybi
from sympy.integrals.integrals import Integral
from sympy.simplify.ratsimp import ratsimp
from sympy.testing.pytest import ON_CI, raises, slow, skip, XFAIL
def test_component_division():
    f, g, h, k = symbols('f g h k', cls=Function)
    x = symbols('x')
    funcs = [f(x), g(x), h(x), k(x)]
    eqs1 = [Eq(Derivative(f(x), x), 2 * f(x)), Eq(Derivative(g(x), x), f(x)), Eq(Derivative(h(x), x), h(x)), Eq(Derivative(k(x), x), h(x) ** 4 + k(x))]
    sol1 = [Eq(f(x), 2 * C1 * exp(2 * x)), Eq(g(x), C1 * exp(2 * x) + C2), Eq(h(x), C3 * exp(x)), Eq(k(x), C3 ** 4 * exp(4 * x) / 3 + C4 * exp(x))]
    assert dsolve(eqs1) == sol1
    assert checksysodesol(eqs1, sol1) == (True, [0, 0, 0, 0])
    components1 = {((Eq(Derivative(f(x), x), 2 * f(x)),), (Eq(Derivative(g(x), x), f(x)),)), ((Eq(Derivative(h(x), x), h(x)),), (Eq(Derivative(k(x), x), h(x) ** 4 + k(x)),))}
    eqsdict1 = ({f(x): set(), g(x): {f(x)}, h(x): set(), k(x): {h(x)}}, {f(x): Eq(Derivative(f(x), x), 2 * f(x)), g(x): Eq(Derivative(g(x), x), f(x)), h(x): Eq(Derivative(h(x), x), h(x)), k(x): Eq(Derivative(k(x), x), h(x) ** 4 + k(x))})
    graph1 = [{f(x), g(x), h(x), k(x)}, {(g(x), f(x)), (k(x), h(x))}]
    assert {tuple((tuple(scc) for scc in wcc)) for wcc in _component_division(eqs1, funcs, x)} == components1
    assert _eqs2dict(eqs1, funcs) == eqsdict1
    assert [set(element) for element in _dict2graph(eqsdict1[0])] == graph1
    eqs2 = [Eq(Derivative(f(x), x), 2 * f(x)), Eq(Derivative(g(x), x), f(x)), Eq(Derivative(h(x), x), h(x)), Eq(Derivative(k(x), x), f(x) ** 4 + k(x))]
    sol2 = [Eq(f(x), C1 * exp(2 * x)), Eq(g(x), C1 * exp(2 * x) / 2 + C2), Eq(h(x), C3 * exp(x)), Eq(k(x), C1 ** 4 * exp(8 * x) / 7 + C4 * exp(x))]
    assert dsolve(eqs2) == sol2
    assert checksysodesol(eqs2, sol2) == (True, [0, 0, 0, 0])
    components2 = {frozenset([(Eq(Derivative(f(x), x), 2 * f(x)),), (Eq(Derivative(g(x), x), f(x)),), (Eq(Derivative(k(x), x), f(x) ** 4 + k(x)),)]), frozenset([(Eq(Derivative(h(x), x), h(x)),)])}
    eqsdict2 = ({f(x): set(), g(x): {f(x)}, h(x): set(), k(x): {f(x)}}, {f(x): Eq(Derivative(f(x), x), 2 * f(x)), g(x): Eq(Derivative(g(x), x), f(x)), h(x): Eq(Derivative(h(x), x), h(x)), k(x): Eq(Derivative(k(x), x), f(x) ** 4 + k(x))})
    graph2 = [{f(x), g(x), h(x), k(x)}, {(g(x), f(x)), (k(x), f(x))}]
    assert {frozenset((tuple(scc) for scc in wcc)) for wcc in _component_division(eqs2, funcs, x)} == components2
    assert _eqs2dict(eqs2, funcs) == eqsdict2
    assert [set(element) for element in _dict2graph(eqsdict2[0])] == graph2
    eqs3 = [Eq(Derivative(f(x), x), 2 * f(x)), Eq(Derivative(g(x), x), x + f(x)), Eq(Derivative(h(x), x), h(x)), Eq(Derivative(k(x), x), f(x) ** 4 + k(x))]
    sol3 = [Eq(f(x), C1 * exp(2 * x)), Eq(g(x), C1 * exp(2 * x) / 2 + C2 + x ** 2 / 2), Eq(h(x), C3 * exp(x)), Eq(k(x), C1 ** 4 * exp(8 * x) / 7 + C4 * exp(x))]
    assert dsolve(eqs3) == sol3
    assert checksysodesol(eqs3, sol3) == (True, [0, 0, 0, 0])
    components3 = {frozenset([(Eq(Derivative(f(x), x), 2 * f(x)),), (Eq(Derivative(g(x), x), x + f(x)),), (Eq(Derivative(k(x), x), f(x) ** 4 + k(x)),)]), frozenset([(Eq(Derivative(h(x), x), h(x)),)])}
    eqsdict3 = ({f(x): set(), g(x): {f(x)}, h(x): set(), k(x): {f(x)}}, {f(x): Eq(Derivative(f(x), x), 2 * f(x)), g(x): Eq(Derivative(g(x), x), x + f(x)), h(x): Eq(Derivative(h(x), x), h(x)), k(x): Eq(Derivative(k(x), x), f(x) ** 4 + k(x))})
    graph3 = [{f(x), g(x), h(x), k(x)}, {(g(x), f(x)), (k(x), f(x))}]
    assert {frozenset((tuple(scc) for scc in wcc)) for wcc in _component_division(eqs3, funcs, x)} == components3
    assert _eqs2dict(eqs3, funcs) == eqsdict3
    assert [set(l) for l in _dict2graph(eqsdict3[0])] == graph3
    eqs4 = [Eq(Derivative(f(x), x), x * f(x) + 2 * g(x)), Eq(Derivative(g(x), x), f(x) + x * g(x) + x), Eq(Derivative(h(x), x), h(x)), Eq(Derivative(k(x), x), f(x) ** 4 + k(x))]
    sol4 = [Eq(f(x), (C1 / 2 - sqrt(2) * C2 / 2 - sqrt(2) * Integral(x * exp(-x ** 2 / 2 - sqrt(2) * x) / 2 + x * exp(-x ** 2 / 2 + sqrt(2) * x) / 2, x) / 2 + Integral(sqrt(2) * x * exp(-x ** 2 / 2 - sqrt(2) * x) / 2 - sqrt(2) * x * exp(-x ** 2 / 2 + sqrt(2) * x) / 2, x) / 2) * exp(x ** 2 / 2 - sqrt(2) * x) + (C1 / 2 + sqrt(2) * C2 / 2 + sqrt(2) * Integral(x * exp(-x ** 2 / 2 - sqrt(2) * x) / 2 + x * exp(-x ** 2 / 2 + sqrt(2) * x) / 2, x) / 2 + Integral(sqrt(2) * x * exp(-x ** 2 / 2 - sqrt(2) * x) / 2 - sqrt(2) * x * exp(-x ** 2 / 2 + sqrt(2) * x) / 2, x) / 2) * exp(x ** 2 / 2 + sqrt(2) * x)), Eq(g(x), (-sqrt(2) * C1 / 4 + C2 / 2 + Integral(x * exp(-x ** 2 / 2 - sqrt(2) * x) / 2 + x * exp(-x ** 2 / 2 + sqrt(2) * x) / 2, x) / 2 - sqrt(2) * Integral(sqrt(2) * x * exp(-x ** 2 / 2 - sqrt(2) * x) / 2 - sqrt(2) * x * exp(-x ** 2 / 2 + sqrt(2) * x) / 2, x) / 4) * exp(x ** 2 / 2 - sqrt(2) * x) + (sqrt(2) * C1 / 4 + C2 / 2 + Integral(x * exp(-x ** 2 / 2 - sqrt(2) * x) / 2 + x * exp(-x ** 2 / 2 + sqrt(2) * x) / 2, x) / 2 + sqrt(2) * Integral(sqrt(2) * x * exp(-x ** 2 / 2 - sqrt(2) * x) / 2 - sqrt(2) * x * exp(-x ** 2 / 2 + sqrt(2) * x) / 2, x) / 4) * exp(x ** 2 / 2 + sqrt(2) * x)), Eq(h(x), C3 * exp(x)), Eq(k(x), C4 * exp(x) + exp(x) * Integral((C1 * exp(x ** 2 / 2 - sqrt(2) * x) / 2 + C1 * exp(x ** 2 / 2 + sqrt(2) * x) / 2 - sqrt(2) * C2 * exp(x ** 2 / 2 - sqrt(2) * x) / 2 + sqrt(2) * C2 * exp(x ** 2 / 2 + sqrt(2) * x) / 2 - sqrt(2) * exp(x ** 2 / 2 - sqrt(2) * x) * Integral(x * exp(-x ** 2 / 2 - sqrt(2) * x) / 2 + x * exp(-x ** 2 / 2 + sqrt(2) * x) / 2, x) / 2 + exp(x ** 2 / 2 - sqrt(2) * x) * Integral(sqrt(2) * x * exp(-x ** 2 / 2 - sqrt(2) * x) / 2 - sqrt(2) * x * exp(-x ** 2 / 2 + sqrt(2) * x) / 2, x) / 2 + sqrt(2) * exp(x ** 2 / 2 + sqrt(2) * x) * Integral(x * exp(-x ** 2 / 2 - sqrt(2) * x) / 2 + x * exp(-x ** 2 / 2 + sqrt(2) * x) / 2, x) / 2 + exp(x ** 2 / 2 + sqrt(2) * x) * Integral(sqrt(2) * x * exp(-x ** 2 / 2 - sqrt(2) * x) / 2 - sqrt(2) * x * exp(-x ** 2 / 2 + sqrt(2) * x) / 2, x) / 2) ** 4 * exp(-x), x))]
    components4 = {(frozenset([Eq(Derivative(f(x), x), x * f(x) + 2 * g(x)), Eq(Derivative(g(x), x), x * g(x) + x + f(x))]), frozenset([Eq(Derivative(k(x), x), f(x) ** 4 + k(x))])), (frozenset([Eq(Derivative(h(x), x), h(x))]),)}
    eqsdict4 = ({f(x): {g(x)}, g(x): {f(x)}, h(x): set(), k(x): {f(x)}}, {f(x): Eq(Derivative(f(x), x), x * f(x) + 2 * g(x)), g(x): Eq(Derivative(g(x), x), x * g(x) + x + f(x)), h(x): Eq(Derivative(h(x), x), h(x)), k(x): Eq(Derivative(k(x), x), f(x) ** 4 + k(x))})
    graph4 = [{f(x), g(x), h(x), k(x)}, {(f(x), g(x)), (g(x), f(x)), (k(x), f(x))}]
    assert {tuple((frozenset(scc) for scc in wcc)) for wcc in _component_division(eqs4, funcs, x)} == components4
    assert _eqs2dict(eqs4, funcs) == eqsdict4
    assert [set(element) for element in _dict2graph(eqsdict4[0])] == graph4
    assert dsolve_system(eqs4, simplify=False, doit=False) == [sol4]
    assert checksysodesol(eqs4, sol4) == (True, [0, 0, 0, 0])
    eqs5 = [Eq(Derivative(f(x), x), x * f(x) + 2 * g(x)), Eq(Derivative(g(x), x), x * g(x) + f(x)), Eq(Derivative(h(x), x), h(x)), Eq(Derivative(k(x), x), f(x) ** 4 + k(x))]
    sol5 = [Eq(f(x), (C1 / 2 - sqrt(2) * C2 / 2) * exp(x ** 2 / 2 - sqrt(2) * x) + (C1 / 2 + sqrt(2) * C2 / 2) * exp(x ** 2 / 2 + sqrt(2) * x)), Eq(g(x), (-sqrt(2) * C1 / 4 + C2 / 2) * exp(x ** 2 / 2 - sqrt(2) * x) + (sqrt(2) * C1 / 4 + C2 / 2) * exp(x ** 2 / 2 + sqrt(2) * x)), Eq(h(x), C3 * exp(x)), Eq(k(x), C4 * exp(x) + exp(x) * Integral((C1 * exp(x ** 2 / 2 - sqrt(2) * x) / 2 + C1 * exp(x ** 2 / 2 + sqrt(2) * x) / 2 - sqrt(2) * C2 * exp(x ** 2 / 2 - sqrt(2) * x) / 2 + sqrt(2) * C2 * exp(x ** 2 / 2 + sqrt(2) * x) / 2) ** 4 * exp(-x), x))]
    components5 = {(frozenset([Eq(Derivative(f(x), x), x * f(x) + 2 * g(x)), Eq(Derivative(g(x), x), x * g(x) + f(x))]), frozenset([Eq(Derivative(k(x), x), f(x) ** 4 + k(x))])), (frozenset([Eq(Derivative(h(x), x), h(x))]),)}
    eqsdict5 = ({f(x): {g(x)}, g(x): {f(x)}, h(x): set(), k(x): {f(x)}}, {f(x): Eq(Derivative(f(x), x), x * f(x) + 2 * g(x)), g(x): Eq(Derivative(g(x), x), x * g(x) + f(x)), h(x): Eq(Derivative(h(x), x), h(x)), k(x): Eq(Derivative(k(x), x), f(x) ** 4 + k(x))})
    graph5 = [{f(x), g(x), h(x), k(x)}, {(f(x), g(x)), (g(x), f(x)), (k(x), f(x))}]
    assert {tuple((frozenset(scc) for scc in wcc)) for wcc in _component_division(eqs5, funcs, x)} == components5
    assert _eqs2dict(eqs5, funcs) == eqsdict5
    assert [set(element) for element in _dict2graph(eqsdict5[0])] == graph5
    assert dsolve_system(eqs5, simplify=False, doit=False) == [sol5]
    assert checksysodesol(eqs5, sol5) == (True, [0, 0, 0, 0])