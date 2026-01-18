import collections
import contextlib
import inspect
import functools
from enum import Enum
from numba.core import typing, types, utils, cgutils
from numba.core.typing.templates import BaseRegistryLoader
def lower_cast(self, fromty, toty):
    """
        Decorate the implementation of implicit conversion between
        *fromty* and *toty*.

        The decorated implementation will have the signature
        (context, builder, fromty, toty, val).
        """

    def decorate(impl):
        self.casts.append((impl, (fromty, toty)))
        return impl
    return decorate