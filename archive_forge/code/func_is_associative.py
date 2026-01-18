from sympy.unify.core import Compound, Variable, CondVariable, allcombinations
from sympy.unify import core
def is_associative(x):
    return isinstance(x, Compound) and x.op in ('Add', 'Mul', 'CAdd', 'CMul')