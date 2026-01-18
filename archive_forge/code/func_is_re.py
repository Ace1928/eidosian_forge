from __future__ import annotations
from collections import abc
from numbers import Number
import re
from re import Pattern
from typing import TYPE_CHECKING
import numpy as np
from pandas._libs import lib
def is_re(obj) -> TypeGuard[Pattern]:
    """
    Check if the object is a regex pattern instance.

    Parameters
    ----------
    obj : The object to check

    Returns
    -------
    bool
        Whether `obj` is a regex pattern.

    Examples
    --------
    >>> from pandas.api.types import is_re
    >>> import re
    >>> is_re(re.compile(".*"))
    True
    >>> is_re("foo")
    False
    """
    return isinstance(obj, Pattern)