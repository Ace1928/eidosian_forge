from sympy.unify.core import Compound, Variable, CondVariable, allcombinations
from sympy.unify import core
def test_CondVariable():
    expr = C('CAdd', (1, 2))
    x = Variable('x')
    y = CondVariable('y', lambda a: a % 2 == 0)
    z = CondVariable('z', lambda a: a > 3)
    pattern = C('CAdd', (x, y))
    assert list(unify(expr, pattern, {})) == [{x: 1, y: 2}]
    z = CondVariable('z', lambda a: a > 3)
    pattern = C('CAdd', (z, y))
    assert list(unify(expr, pattern, {})) == []