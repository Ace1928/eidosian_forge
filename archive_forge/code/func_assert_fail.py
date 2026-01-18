import itertools
import numpy as np
from numba import jit
from numba.core import utils
from numba.tests.support import TestCase, forbid_codegen
from .enum_usecases import *
import unittest
def assert_fail(left, right):
    try:
        self.assertPreciseEqual(left, right, **kwargs)
    except AssertionError:
        pass
    else:
        self.fail('%s and %s unexpectedly considered equal' % (left, right))