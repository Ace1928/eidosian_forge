import contextlib
import copy
import operator
import pickle
import warnings
from absl.testing import absltest
from absl.testing import parameterized
import ml_dtypes
import numpy as np
@parameterized.product(scalar_type=INT4_TYPES, ufunc=[np.nonzero, np.logical_not])
def testUnaryPredicateUfunc(self, scalar_type, ufunc):
    x = np.array(VALUES[scalar_type])
    y = np.array(VALUES[scalar_type], dtype=scalar_type)
    np.testing.assert_array_equal(ufunc(x), ufunc(y))