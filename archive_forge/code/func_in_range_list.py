from __future__ import annotations
import decimal
import re
from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, Any, Callable
def in_range_list(num: float | decimal.Decimal, range_list: Iterable[Iterable[float | decimal.Decimal]]) -> bool:
    """Integer range list test.  This is the callback for the "in" operator
    of the UTS #35 pluralization rule language:

    >>> in_range_list(1, [(1, 3)])
    True
    >>> in_range_list(3, [(1, 3)])
    True
    >>> in_range_list(3, [(1, 3), (5, 8)])
    True
    >>> in_range_list(1.2, [(1, 4)])
    False
    >>> in_range_list(10, [(1, 4)])
    False
    >>> in_range_list(10, [(1, 4), (6, 8)])
    False
    """
    return num == int(num) and within_range_list(num, range_list)