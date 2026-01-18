from __future__ import annotations
from operator import attrgetter
from collections import defaultdict
from sympy.utilities.exceptions import sympy_deprecation_warning
from .sympify import _sympify as _sympify_, sympify
from .basic import Basic
from .cache import cacheit
from .sorting import ordered
from .logic import fuzzy_and
from .parameters import global_parameters
from sympy.utilities.iterables import sift
from sympy.multipledispatch.dispatcher import (Dispatcher,
def register_handlerclass(self, classes, typ, on_ambiguity=ambiguity_register_error_ignore_dup):
    """
        Register the handler class for two classes, in both straight and reversed order.

        Paramteters
        ===========

        classes : tuple of two types
            Classes who are compared with each other.

        typ:
            Class which is registered to represent *cls1* and *cls2*.
            Handler method of *self* must be implemented in this class.
        """
    if not len(classes) == 2:
        raise RuntimeError('Only binary dispatch is supported, but got %s types: <%s>.' % (len(classes), str_signature(classes)))
    if len(set(classes)) == 1:
        raise RuntimeError('Duplicate types <%s> cannot be dispatched.' % str_signature(classes))
    self._dispatcher.add(tuple(classes), typ, on_ambiguity=on_ambiguity)
    self._dispatcher.add(tuple(reversed(classes)), typ, on_ambiguity=on_ambiguity)