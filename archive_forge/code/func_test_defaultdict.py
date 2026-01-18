from sympy.unify.core import Compound, Variable, CondVariable, allcombinations
from sympy.unify import core
def test_defaultdict():
    assert next(unify(Variable('x'), 'foo')) == {Variable('x'): 'foo'}