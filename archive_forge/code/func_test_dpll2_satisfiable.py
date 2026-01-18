from sympy.assumptions.ask import Q
from sympy.core.symbol import symbols
from sympy.logic.boolalg import And, Implies, Equivalent, true, false
from sympy.logic.inference import literal_symbol, \
from sympy.logic.algorithms.dpll import dpll, dpll_satisfiable, \
from sympy.logic.algorithms.dpll2 import dpll_satisfiable as dpll2_satisfiable
from sympy.testing.pytest import raises
def test_dpll2_satisfiable():
    A, B, C = symbols('A,B,C')
    assert dpll2_satisfiable(A & ~A) is False
    assert dpll2_satisfiable(A & ~B) == {A: True, B: False}
    assert dpll2_satisfiable(A | B) in ({A: True}, {B: True}, {A: True, B: True})
    assert dpll2_satisfiable((~A | B) & (~B | A)) in ({A: True, B: True}, {A: False, B: False})
    assert dpll2_satisfiable((A | B) & (~B | C)) in ({A: True, B: False, C: True}, {A: True, B: True, C: True})
    assert dpll2_satisfiable(A & B & C) == {A: True, B: True, C: True}
    assert dpll2_satisfiable((A | B) & A >> B) in ({B: True, A: False}, {B: True, A: True})
    assert dpll2_satisfiable(Equivalent(A, B) & A) == {A: True, B: True}
    assert dpll2_satisfiable(Equivalent(A, B) & ~A) == {A: False, B: False}