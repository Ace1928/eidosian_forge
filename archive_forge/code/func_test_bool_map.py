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
def test_bool_map():
    """
    Test working of bool_map function.
    """
    minterms = [[0, 0, 0, 1], [0, 0, 1, 1], [0, 1, 1, 1], [1, 0, 1, 1], [1, 1, 1, 1]]
    assert bool_map(Not(Not(a)), a) == (a, {a: a})
    assert bool_map(SOPform([w, x, y, z], minterms), POSform([w, x, y, z], minterms)) == (And(Or(Not(w), y), Or(Not(x), y), z), {x: x, w: w, z: z, y: y})
    assert bool_map(SOPform([x, z, y], [[1, 0, 1]]), SOPform([a, b, c], [[1, 0, 1]])) != False
    function1 = SOPform([x, z, y], [[1, 0, 1], [0, 0, 1]])
    function2 = SOPform([a, b, c], [[1, 0, 1], [1, 0, 0]])
    assert bool_map(function1, function2) == (function1, {y: a, z: b})
    assert bool_map(Xor(x, y), ~Xor(x, y)) == False
    assert bool_map(And(x, y), Or(x, y)) is None
    assert bool_map(And(x, y), And(x, y, z)) is None
    assert bool_map(Xor(x, y, z), ~Xor(x, y, z)) == False
    assert bool_map(Xor(a, x, y, z), ~Xor(a, x, y, z)) == False