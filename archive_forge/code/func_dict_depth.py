from __future__ import annotations
import datetime
import inspect
import logging
import re
import sys
import typing as t
from collections.abc import Collection, Set
from contextlib import contextmanager
from copy import copy
from enum import Enum
from itertools import count
def dict_depth(d: t.Dict) -> int:
    """
    Get the nesting depth of a dictionary.

    Example:
        >>> dict_depth(None)
        0
        >>> dict_depth({})
        1
        >>> dict_depth({"a": "b"})
        1
        >>> dict_depth({"a": {}})
        2
        >>> dict_depth({"a": {"b": {}}})
        3
    """
    try:
        return 1 + dict_depth(next(iter(d.values())))
    except AttributeError:
        return 0
    except StopIteration:
        return 1