import contextlib
import copy
import enum
import functools
import inspect
import itertools
import linecache
import sys
import types
import typing
from operator import itemgetter
from . import _compat, _config, setters
from ._compat import (
from .exceptions import (
def fields_dict(cls):
    """
    Return an ordered dictionary of *attrs* attributes for a class, whose
    keys are the attribute names.

    :param type cls: Class to introspect.

    :raise TypeError: If *cls* is not a class.
    :raise attrs.exceptions.NotAnAttrsClassError: If *cls* is not an *attrs*
        class.

    :rtype: dict

    .. versionadded:: 18.1.0
    """
    if not isinstance(cls, type):
        msg = 'Passed object must be a class.'
        raise TypeError(msg)
    attrs = getattr(cls, '__attrs_attrs__', None)
    if attrs is None:
        msg = f'{cls!r} is not an attrs-decorated class.'
        raise NotAnAttrsClassError(msg)
    return {a.name: a for a in attrs}