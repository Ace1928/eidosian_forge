from __future__ import annotations
import operator
import sys
import os
import re as _re
import struct
from textwrap import fill, dedent
def func_name(x, short=False):
    """Return function name of `x` (if defined) else the `type(x)`.
    If short is True and there is a shorter alias for the result,
    return the alias.

    Examples
    ========

    >>> from sympy.utilities.misc import func_name
    >>> from sympy import Matrix
    >>> from sympy.abc import x
    >>> func_name(Matrix.eye(3))
    'MutableDenseMatrix'
    >>> func_name(x < 1)
    'StrictLessThan'
    >>> func_name(x < 1, short=True)
    'Lt'
    """
    alias = {'GreaterThan': 'Ge', 'StrictGreaterThan': 'Gt', 'LessThan': 'Le', 'StrictLessThan': 'Lt', 'Equality': 'Eq', 'Unequality': 'Ne'}
    typ = type(x)
    if str(typ).startswith("<type '"):
        typ = str(typ).split("'")[1].split("'")[0]
    elif str(typ).startswith("<class '"):
        typ = str(typ).split("'")[1].split("'")[0]
    rv = getattr(getattr(x, 'func', x), '__name__', typ)
    if '.' in rv:
        rv = rv.split('.')[-1]
    if short:
        rv = alias.get(rv, rv)
    return rv