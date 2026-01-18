from __future__ import annotations
from collections.abc import Iterable
from typing import Final, TYPE_CHECKING, Callable
import numpy as np
def get_type_analyze_hook(self, fullname: str) -> None | _HookFunc:
    """Set the precision of platform-specific `numpy.number`
            subclasses.

            For example: `numpy.int_`, `numpy.longlong` and `numpy.longdouble`.
            """
    if fullname in _PRECISION_DICT:
        return _hook
    return None