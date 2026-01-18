import collections
import contextlib
import inspect
import functools
from enum import Enum
from numba.core import typing, types, utils, cgutils
from numba.core.typing.templates import BaseRegistryLoader
def set_valid(self, is_valid=True):
    """
        Mark the iterator as valid according to *is_valid* (which must
        be either a Python boolean or a LLVM inst).
        """
    if is_valid in (False, True):
        is_valid = self._context.get_constant(types.boolean, is_valid)
    self._pairobj.second = is_valid