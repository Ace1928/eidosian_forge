from sympy.core import Basic, Add, Mul, Pow
from sympy.core.operations import AssocOp, LatticeOp
from sympy.matrices import MatAdd, MatMul, MatrixExpr
from sympy.sets.sets import Union, Intersection, FiniteSet
from sympy.unify.core import Compound, Variable, CondVariable
from sympy.unify import core
def sympy_associative(op):
    assoc_ops = (AssocOp, MatAdd, MatMul, Union, Intersection, FiniteSet)
    return any((issubclass(op, aop) for aop in assoc_ops))