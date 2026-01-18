import unittest
from contextlib import contextmanager
from functools import cached_property
from numba import njit
from numba.core import errors, cpu, typing
from numba.core.descriptors import TargetDescriptor
from numba.core.dispatcher import TargetConfigurationStack
from numba.core.retarget import BasicRetarget
from numba.core.extending import overload
from numba.core.target_extension import (
@overload(overloaded_func, target=CUSTOM_TARGET)
def ol_overloaded_func_custom_target(x):

    def impl(x):
        return 62830
    return impl