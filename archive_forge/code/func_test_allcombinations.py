from sympy.unify.core import Compound, Variable, CondVariable, allcombinations
from sympy.unify import core
def test_allcombinations():
    assert set(allcombinations((1, 2), (1, 2), 'commutative')) == {(((1,), (2,)), ((1,), (2,))), (((1,), (2,)), ((2,), (1,)))}