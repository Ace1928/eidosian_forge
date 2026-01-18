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
def testAddScalarTypePromotion(self, float_type):
    """Tests type promotion against Numpy scalar values."""
    types = [float_type, np.float16, np.float32, np.float64, np.longdouble]
    for lhs_type in types:
        for rhs_type in types:
            expected_type = numpy_promote_types(lhs_type, rhs_type, float_type=float_type, next_largest_fp_type=np.float32)
            actual_type = type(lhs_type(3.5) + rhs_type(2.25))
            self.assertEqual(expected_type, actual_type)