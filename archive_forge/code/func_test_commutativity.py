from sympy.unify.core import Compound, Variable, CondVariable, allcombinations
from sympy.unify import core
def test_commutativity():
    c1 = Compound('CAdd', (a, b))
    c2 = Compound('CAdd', (x, y))
    assert is_commutative(c1) and is_commutative(c2)
    assert len(list(unify(c1, c2, {}))) == 2