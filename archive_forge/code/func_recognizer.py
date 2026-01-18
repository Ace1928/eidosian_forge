from . import z3core
from .z3core import *
from .z3types import *
from .z3consts import *
from .z3printer import *
from fractions import Fraction
import sys
import io
import math
import copy
def recognizer(self, idx):
    """In Z3, each constructor has an associated recognizer predicate.

        If the constructor is named `name`, then the recognizer `is_name`.

        >>> List = Datatype('List')
        >>> List.declare('cons', ('car', IntSort()), ('cdr', List))
        >>> List.declare('nil')
        >>> List = List.create()
        >>> # List is now a Z3 declaration
        >>> List.num_constructors()
        2
        >>> List.recognizer(0)
        is(cons)
        >>> List.recognizer(1)
        is(nil)
        >>> simplify(List.is_nil(List.cons(10, List.nil)))
        False
        >>> simplify(List.is_cons(List.cons(10, List.nil)))
        True
        >>> l = Const('l', List)
        >>> simplify(List.is_cons(l))
        is(cons, l)
        """
    if z3_debug():
        _z3_assert(idx < self.num_constructors(), 'Invalid recognizer index')
    return FuncDeclRef(Z3_get_datatype_sort_recognizer(self.ctx_ref(), self.ast, idx), self.ctx)