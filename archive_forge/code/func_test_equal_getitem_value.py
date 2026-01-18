import numpy as np
import unittest
from numba import jit, from_dtype
from numba.core import types
from numba.typed import Dict
from numba.tests.support import (TestCase, skip_ppc64le_issue4563)
def test_equal_getitem_value(self):
    self._test_op_getitem_value(equal_getitem_value)