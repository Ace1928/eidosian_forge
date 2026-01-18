import collections
import contextlib
import inspect
import functools
from enum import Enum
from numba.core import typing, types, utils, cgutils
from numba.core.typing.templates import BaseRegistryLoader
def set_exhausted(self):
    """
        Mark the iterator as exhausted.
        """
    self._pairobj.second = self._context.get_constant(types.boolean, False)