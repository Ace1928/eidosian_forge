from __future__ import print_function, unicode_literals
import typing
import re
from functools import partial
from .lrucache import LRUCache
def imatch_any(patterns, name):
    """Test if a name matches any of a list of patterns (case insensitive).

    Will return `True` if ``patterns`` is an empty list.

    Arguments:
        patterns (list): A list of wildcard pattern, e.g ``["*.py",
            "*.pyc"]``
        name (str): A filename.

    Returns:
        bool: `True` if the name matches at least one of the patterns.

    """
    if not patterns:
        return True
    return any((imatch(pattern, name) for pattern in patterns))