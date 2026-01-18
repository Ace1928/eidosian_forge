import numpy as np
from numba.core.utils import PYVERSION
from numba.cuda.testing import skip_on_cudasim, CUDATestCase
from numba.tests.support import (override_config, captured_stderr,
from numba import cuda, float64
import unittest
def assert_fails(self, *args, **kwargs):
    self.assertRaises(AssertionError, *args, **kwargs)