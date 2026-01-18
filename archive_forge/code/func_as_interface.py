from __future__ import annotations
import collections
import enum
from functools import update_wrapper
import inspect
import itertools
import operator
import re
import sys
import textwrap
import threading
import types
from types import CodeType
from typing import Any
from typing import Callable
from typing import cast
from typing import Dict
from typing import FrozenSet
from typing import Generic
from typing import Iterator
from typing import List
from typing import Mapping
from typing import NoReturn
from typing import Optional
from typing import overload
from typing import Sequence
from typing import Set
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
import warnings
from . import _collections
from . import compat
from ._has_cy import HAS_CYEXTENSION
from .typing import Literal
from .. import exc
def as_interface(obj, cls=None, methods=None, required=None):
    """Ensure basic interface compliance for an instance or dict of callables.

    Checks that ``obj`` implements public methods of ``cls`` or has members
    listed in ``methods``. If ``required`` is not supplied, implementing at
    least one interface method is sufficient. Methods present on ``obj`` that
    are not in the interface are ignored.

    If ``obj`` is a dict and ``dict`` does not meet the interface
    requirements, the keys of the dictionary are inspected. Keys present in
    ``obj`` that are not in the interface will raise TypeErrors.

    Raises TypeError if ``obj`` does not meet the interface criteria.

    In all passing cases, an object with callable members is returned.  In the
    simple case, ``obj`` is returned as-is; if dict processing kicks in then
    an anonymous class is returned.

    obj
      A type, instance, or dictionary of callables.
    cls
      Optional, a type.  All public methods of cls are considered the
      interface.  An ``obj`` instance of cls will always pass, ignoring
      ``required``..
    methods
      Optional, a sequence of method names to consider as the interface.
    required
      Optional, a sequence of mandatory implementations. If omitted, an
      ``obj`` that provides at least one interface method is considered
      sufficient.  As a convenience, required may be a type, in which case
      all public methods of the type are required.

    """
    if not cls and (not methods):
        raise TypeError('a class or collection of method names are required')
    if isinstance(cls, type) and isinstance(obj, cls):
        return obj
    interface = set(methods or [m for m in dir(cls) if not m.startswith('_')])
    implemented = set(dir(obj))
    complies = operator.ge
    if isinstance(required, type):
        required = interface
    elif not required:
        required = set()
        complies = operator.gt
    else:
        required = set(required)
    if complies(implemented.intersection(interface), required):
        return obj
    if not isinstance(obj, dict):
        qualifier = complies is operator.gt and 'any of' or 'all of'
        raise TypeError('%r does not implement %s: %s' % (obj, qualifier, ', '.join(interface)))

    class AnonymousInterface:
        """A callable-holding shell."""
    if cls:
        AnonymousInterface.__name__ = 'Anonymous' + cls.__name__
    found = set()
    for method, impl in dictlike_iteritems(obj):
        if method not in interface:
            raise TypeError('%r: unknown in this interface' % method)
        if not callable(impl):
            raise TypeError('%r=%r is not callable' % (method, impl))
        setattr(AnonymousInterface, method, staticmethod(impl))
        found.add(method)
    if complies(found, required):
        return AnonymousInterface
    raise TypeError('dictionary does not contain required keys %s' % ', '.join(required - found))