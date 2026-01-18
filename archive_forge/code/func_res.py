import collections
import contextlib
import inspect
import functools
from enum import Enum
from numba.core import typing, types, utils, cgutils
from numba.core.typing.templates import BaseRegistryLoader
def res(context, builder, sig, args, attr):
    return real_impl(context, builder, sig, args, attr)