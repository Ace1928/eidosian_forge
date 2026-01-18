import collections
import contextlib
import copy
import itertools
import math
import pickle
import sys
from typing import Type
import warnings
from absl.testing import absltest
from absl.testing import parameterized
import ml_dtypes
import numpy as np
def testArray(self, float_type):
    x = np.array([[1, 2, 3]], dtype=float_type)
    self.assertEqual(float_type, x.dtype)
    self.assertEqual('[[1 2 3]]', str(x))
    np.testing.assert_equal(x, x)
    numpy_assert_allclose(x, x, float_type=float_type)
    self.assertTrue((x == x).all())