from __future__ import annotations
from sympy.core.basic import Basic
from sympy.core.containers import Tuple
from sympy.core.evalf import EvalfMixin, N
from sympy.core.numbers import oo
from sympy.core.symbol import Dummy
from sympy.core.sympify import sympify
from sympy.functions.elementary.trigonometric import cos, sin, atan
from sympy.matrices import eye
from sympy.multipledispatch import dispatch
from sympy.printing import sstr
from sympy.sets import Set, Union, FiniteSet
from sympy.sets.handlers.intersection import intersection_sets
from sympy.sets.handlers.union import union_sets
from sympy.solvers.solvers import solve
from sympy.utilities.misc import func_name
from sympy.utilities.iterables import is_sequence
def is_seq_and_not_point(a):
    if hasattr(a, 'is_Point') and a.is_Point:
        return False
    return is_sequence(a)