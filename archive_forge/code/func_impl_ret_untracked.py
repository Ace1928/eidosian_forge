import collections
import contextlib
import inspect
import functools
from enum import Enum
from numba.core import typing, types, utils, cgutils
from numba.core.typing.templates import BaseRegistryLoader
def impl_ret_untracked(ctx, builder, retty, ret):
    """
    The return type is not a NRT object.
    """
    return ret