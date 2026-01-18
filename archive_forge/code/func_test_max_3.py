import itertools
import functools
import sys
import operator
from collections import namedtuple
import numpy as np
import unittest
import warnings
from numba import jit, typeof, njit, typed
from numba.core import errors, types, config
from numba.tests.support import (TestCase, tag, ignore_internal_warnings,
from numba.core.extending import overload_method, box
def test_max_3(self, flags=forceobj_flags):
    """
        max(tuple)
        """
    self.check_minmax_3(max_usecase3, flags)