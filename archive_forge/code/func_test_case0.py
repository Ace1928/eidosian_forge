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
def test_case0(self):
    fixed_target = self.functions['fixed_target']
    flex_target = self.functions['flex_target']

    @njit
    def foo(x):
        x = fixed_target(x)
        x = flex_target(x)
        return x
    r = foo(123)
    self.assertEqual(r, 123 + 10 + 1000)
    stats = self.retarget.cache.stats()
    self.assertEqual(stats, dict(hit=0, miss=0))