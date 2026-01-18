import collections
import contextlib
import inspect
import functools
from enum import Enum
from numba.core import typing, types, utils, cgutils
from numba.core.typing.templates import BaseRegistryLoader
def lower_setattr(self, ty, attr):
    """
        Decorate an implementation of __setattr__ for type *ty* and
        the attribute *attr*.

        The decorated implementation will have the signature
        (context, builder, sig, args).
        """

    def decorate(impl):
        return self._decorate_attr(impl, ty, attr, self.setattrs, _decorate_setattr)
    return decorate