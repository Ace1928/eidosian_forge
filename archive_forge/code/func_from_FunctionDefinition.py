from __future__ import annotations
from typing import Any
from collections import defaultdict
from sympy.core.relational import (Ge, Gt, Le, Lt)
from sympy.core import Symbol, Tuple, Dummy
from sympy.core.basic import Basic
from sympy.core.expr import Expr, Atom
from sympy.core.numbers import Float, Integer, oo
from sympy.core.sympify import _sympify, sympify, SympifyError
from sympy.utilities.iterables import (iterable, topological_sort,
@classmethod
def from_FunctionDefinition(cls, func_def):
    if not isinstance(func_def, FunctionDefinition):
        raise TypeError('func_def is not an instance of FunctionDefinition')
    return cls(**func_def.kwargs(exclude=('body',)))