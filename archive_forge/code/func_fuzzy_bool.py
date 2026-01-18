from __future__ import annotations
from typing import Optional
def fuzzy_bool(x):
    """Return True, False or None according to x.

    Whereas bool(x) returns True or False, fuzzy_bool allows
    for the None value and non-false values (which become None), too.

    Examples
    ========

    >>> from sympy.core.logic import fuzzy_bool
    >>> from sympy.abc import x
    >>> fuzzy_bool(x), fuzzy_bool(None)
    (None, None)
    >>> bool(x), bool(None)
    (True, False)

    """
    if x is None:
        return None
    if x in (True, False):
        return bool(x)