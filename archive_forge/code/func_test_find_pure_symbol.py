from sympy.assumptions.ask import Q
from sympy.core.symbol import symbols
from sympy.logic.boolalg import And, Implies, Equivalent, true, false
from sympy.logic.inference import literal_symbol, \
from sympy.logic.algorithms.dpll import dpll, dpll_satisfiable, \
from sympy.logic.algorithms.dpll2 import dpll_satisfiable as dpll2_satisfiable
from sympy.testing.pytest import raises
def test_find_pure_symbol():
    A, B, C = symbols('A,B,C')
    assert find_pure_symbol([A], [A]) == (A, True)
    assert find_pure_symbol([A, B], [~A | B, ~B | A]) == (None, None)
    assert find_pure_symbol([A, B, C], [A | ~B, ~B | ~C, C | A]) == (A, True)
    assert find_pure_symbol([A, B, C], [~A | B, B | ~C, C | A]) == (B, True)
    assert find_pure_symbol([A, B, C], [~A | ~B, ~B | ~C, C | A]) == (B, False)
    assert find_pure_symbol([A, B, C], [~A | B, ~B | ~C, C | A]) == (None, None)