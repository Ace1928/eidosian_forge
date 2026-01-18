from sympy.assumptions.ask import Q
from sympy.core.symbol import symbols
from sympy.logic.boolalg import And, Implies, Equivalent, true, false
from sympy.logic.inference import literal_symbol, \
from sympy.logic.algorithms.dpll import dpll, dpll_satisfiable, \
from sympy.logic.algorithms.dpll2 import dpll_satisfiable as dpll2_satisfiable
from sympy.testing.pytest import raises
def test_entails():
    A, B, C = symbols('A, B, C')
    assert entails(A, [A >> B, ~B]) is False
    assert entails(B, [Equivalent(A, B), A]) is True
    assert entails(A >> B >> (~A >> ~B)) is False
    assert entails(A >> B >> (~B >> ~A)) is True