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
def from_FunctionPrototype(cls, func_proto, body):
    if not isinstance(func_proto, FunctionPrototype):
        raise TypeError('func_proto is not an instance of FunctionPrototype')
    return cls(body=body, **func_proto.kwargs())