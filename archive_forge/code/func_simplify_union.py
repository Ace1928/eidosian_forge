from typing import Any, Callable
from functools import reduce
from collections import defaultdict
import inspect
from sympy.core.kind import Kind, UndefinedKind, NumberKind
from sympy.core.basic import Basic
from sympy.core.containers import Tuple, TupleKind
from sympy.core.decorators import sympify_method_args, sympify_return
from sympy.core.evalf import EvalfMixin
from sympy.core.expr import Expr
from sympy.core.function import Lambda
from sympy.core.logic import (FuzzyBool, fuzzy_bool, fuzzy_or, fuzzy_and,
from sympy.core.numbers import Float, Integer
from sympy.core.operations import LatticeOp
from sympy.core.parameters import global_parameters
from sympy.core.relational import Eq, Ne, is_lt
from sympy.core.singleton import Singleton, S
from sympy.core.sorting import ordered
from sympy.core.symbol import symbols, Symbol, Dummy, uniquely_named_symbol
from sympy.core.sympify import _sympify, sympify, _sympy_converter
from sympy.functions.elementary.exponential import exp, log
from sympy.functions.elementary.miscellaneous import Max, Min
from sympy.logic.boolalg import And, Or, Not, Xor, true, false
from sympy.utilities.decorator import deprecated
from sympy.utilities.exceptions import sympy_deprecation_warning
from sympy.utilities.iterables import (iproduct, sift, roundrobin, iterable,
from sympy.utilities.misc import func_name, filldedent
from mpmath import mpi, mpf
from mpmath.libmp.libmpf import prec_to_dps
def simplify_union(args):
    """
    Simplify a :class:`Union` using known rules.

    Explanation
    ===========

    We first start with global rules like 'Merge all FiniteSets'

    Then we iterate through all pairs and ask the constituent sets if they
    can simplify themselves with any other constituent.  This process depends
    on ``union_sets(a, b)`` functions.
    """
    from sympy.sets.handlers.union import union_sets
    if not args:
        return S.EmptySet
    for arg in args:
        if not isinstance(arg, Set):
            raise TypeError('Input args to Union must be Sets')
    finite_sets = [x for x in args if x.is_FiniteSet]
    if len(finite_sets) > 1:
        a = (x for set in finite_sets for x in set)
        finite_set = FiniteSet(*a)
        args = [finite_set] + [x for x in args if not x.is_FiniteSet]
    args = set(args)
    new_args = True
    while new_args:
        for s in args:
            new_args = False
            for t in args - {s}:
                new_set = union_sets(s, t)
                if new_set is not None:
                    if not isinstance(new_set, set):
                        new_set = {new_set}
                    new_args = (args - {s, t}).union(new_set)
                    break
            if new_args:
                args = new_args
                break
    if len(args) == 1:
        return args.pop()
    else:
        return Union(*args, evaluate=False)