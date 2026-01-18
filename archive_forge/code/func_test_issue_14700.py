from sympy.assumptions.ask import Q
from sympy.assumptions.refine import refine
from sympy.core.numbers import oo
from sympy.core.relational import Equality, Eq, Ne
from sympy.core.singleton import S
from sympy.core.symbol import (Dummy, symbols)
from sympy.functions import Piecewise
from sympy.functions.elementary.trigonometric import cos, sin
from sympy.sets.sets import (Interval, Union)
from sympy.simplify.simplify import simplify
from sympy.logic.boolalg import (
from sympy.assumptions.cnf import CNF
from sympy.testing.pytest import raises, XFAIL, slow
from itertools import combinations, permutations, product
def test_issue_14700():
    A, B, C, D, E, F, G, H = symbols('A B C D E F G H')
    q = B & D & H & ~F | B & H & ~C & ~D | B & H & ~C & ~F | B & H & ~D & ~G | B & H & ~F & ~G | C & G & ~B & ~D | C & G & ~D & ~H | C & G & ~F & ~H | D & F & H & ~B | D & F & ~G & ~H | B & D & F & ~C & ~H | D & E & F & ~B & ~C | D & F & ~A & ~B & ~C | D & F & ~A & ~C & ~H | A & B & D & F & ~E & ~H
    soldnf = B & D & H & ~F | D & F & H & ~B | B & H & ~C & ~D | B & H & ~D & ~G | C & G & ~B & ~D | C & G & ~D & ~H | C & G & ~F & ~H | D & F & ~G & ~H | D & E & F & ~C & ~H | D & F & ~A & ~C & ~H | A & B & D & F & ~E & ~H
    solcnf = (B | C | D) & (B | D | G) & (C | D | H) & (C | F | H) & (D | G | H) & (F | G | H) & (B | F | ~D | ~H) & (~B | ~D | ~F | ~H) & (D | ~B | ~C | ~G | ~H) & (A | H | ~C | ~D | ~F | ~G) & (H | ~C | ~D | ~E | ~F | ~G) & (B | E | H | ~A | ~D | ~F | ~G)
    assert simplify_logic(q, 'dnf') == soldnf
    assert simplify_logic(q, 'cnf') == solcnf
    minterms = [[0, 1, 0, 0], [0, 1, 0, 1], [0, 1, 1, 0], [0, 1, 1, 1], [0, 0, 1, 1], [1, 0, 1, 1]]
    dontcares = [[1, 0, 0, 0], [1, 0, 0, 1], [1, 1, 0, 0], [1, 1, 0, 1]]
    assert SOPform([w, x, y, z], minterms) == x & ~w | y & z & ~x
    assert SOPform([w, x, y, z], minterms, dontcares) == x & ~w | y & z & ~x