from __future__ import annotations
from collections import defaultdict
from collections.abc import Mapping
from itertools import chain, zip_longest
from .assumptions import _prepare_class_assumptions
from .cache import cacheit
from .core import ordering_of_classes
from .sympify import _sympify, sympify, SympifyError, _external_converter
from .sorting import ordered
from .kind import Kind, UndefinedKind
from ._print_helpers import Printable
from sympy.utilities.decorator import deprecated
from sympy.utilities.exceptions import sympy_deprecation_warning
from sympy.utilities.iterables import iterable, numbered_symbols
from sympy.utilities.misc import filldedent, func_name
from inspect import getmro
from .singleton import S
from .traversal import (preorder_traversal as _preorder_traversal,
def has_xfree(self, s: set[Basic]):
    """Return True if self has any of the patterns in s as a
        free argument, else False. This is like `Basic.has_free`
        but this will only report exact argument matches.

        Examples
        ========

        >>> from sympy import Function
        >>> from sympy.abc import x, y
        >>> f = Function('f')
        >>> f(x).has_xfree({f})
        False
        >>> f(x).has_xfree({f(x)})
        True
        >>> f(x + 1).has_xfree({x})
        True
        >>> f(x + 1).has_xfree({x + 1})
        True
        >>> f(x + y + 1).has_xfree({x + 1})
        False
        """
    if type(s) is not set:
        raise TypeError('expecting set argument')
    return any((a in s for a in iterfreeargs(self)))