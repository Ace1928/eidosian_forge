from sympy.core import Function, S, sympify, NumberKind
from sympy.utilities.iterables import sift
from sympy.core.add import Add
from sympy.core.containers import Tuple
from sympy.core.operations import LatticeOp, ShortCircuit
from sympy.core.function import (Application, Lambda,
from sympy.core.expr import Expr
from sympy.core.exprtools import factor_terms
from sympy.core.mod import Mod
from sympy.core.mul import Mul
from sympy.core.numbers import Rational
from sympy.core.power import Pow
from sympy.core.relational import Eq, Relational
from sympy.core.singleton import Singleton
from sympy.core.sorting import ordered
from sympy.core.symbol import Dummy
from sympy.core.rules import Transform
from sympy.core.logic import fuzzy_and, fuzzy_or, _torf
from sympy.core.traversal import walk
from sympy.core.numbers import Integer
from sympy.logic.boolalg import And, Or
def factor_minmax(args):
    is_other = lambda arg: isinstance(arg, other)
    other_args, remaining_args = sift(args, is_other, binary=True)
    if not other_args:
        return args
    arg_sets = [set(arg.args) for arg in other_args]
    common = set.intersection(*arg_sets)
    if not common:
        return args
    new_other_args = list(common)
    arg_sets_diff = [arg_set - common for arg_set in arg_sets]
    if all(arg_sets_diff):
        other_args_diff = [other(*s, evaluate=False) for s in arg_sets_diff]
        new_other_args.append(cls(*other_args_diff, evaluate=False))
    other_args_factored = other(*new_other_args, evaluate=False)
    return remaining_args + [other_args_factored]