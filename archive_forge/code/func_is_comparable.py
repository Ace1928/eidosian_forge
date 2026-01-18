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
def is_comparable(self):
    """Return True if self can be computed to a real number
        (or already is a real number) with precision, else False.

        Examples
        ========

        >>> from sympy import exp_polar, pi, I
        >>> (I*exp_polar(I*pi/2)).is_comparable
        True
        >>> (I*exp_polar(I*pi*2)).is_comparable
        False

        A False result does not mean that `self` cannot be rewritten
        into a form that would be comparable. For example, the
        difference computed below is zero but without simplification
        it does not evaluate to a zero with precision:

        >>> e = 2**pi*(1 + 2**pi)
        >>> dif = e - e.expand()
        >>> dif.is_comparable
        False
        >>> dif.n(2)._prec
        1

        """
    is_extended_real = self.is_extended_real
    if is_extended_real is False:
        return False
    if not self.is_number:
        return False
    n, i = [p.evalf(2) if not p.is_Number else p for p in self.as_real_imag()]
    if not (i.is_Number and n.is_Number):
        return False
    if i:
        return False
    else:
        return n._prec != 1