import collections
import contextlib
import inspect
import functools
from enum import Enum
from numba.core import typing, types, utils, cgutils
from numba.core.typing.templates import BaseRegistryLoader
def lower_getattr(self, ty, attr):
    """
        Decorate an implementation of __getattr__ for type *ty* and
        the attribute *attr*.

        The decorated implementation will have the signature
        (context, builder, typ, val).
        """

    def decorate(impl):
        return self._decorate_attr(impl, ty, attr, self.getattrs, _decorate_getattr)
    return decorate