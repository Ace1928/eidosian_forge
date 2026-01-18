from sympy.logic.utilities.dimacs import load
from sympy.logic.algorithms.dpll import dpll_satisfiable
def test_f5():
    assert bool(dpll_satisfiable(load(f5)))