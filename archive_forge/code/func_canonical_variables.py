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
@property
def canonical_variables(self):
    """Return a dictionary mapping any variable defined in
        ``self.bound_symbols`` to Symbols that do not clash
        with any free symbols in the expression.

        Examples
        ========

        >>> from sympy import Lambda
        >>> from sympy.abc import x
        >>> Lambda(x, 2*x).canonical_variables
        {x: _0}
        """
    if not hasattr(self, 'bound_symbols'):
        return {}
    dums = numbered_symbols('_')
    reps = {}
    bound = self.bound_symbols
    names = {i.name for i in self.free_symbols - set(bound)}
    for b in bound:
        d = next(dums)
        if b.is_Symbol:
            while d.name in names:
                d = next(dums)
        reps[b] = d
    return reps