import contextlib
import copy
import operator
import pickle
import warnings
from absl.testing import absltest
from absl.testing import parameterized
import ml_dtypes
import numpy as np
@parameterized.product(scalar_type=INT4_TYPES, ufunc=[np.add, np.subtract, np.multiply, np.floor_divide, np.remainder])
@ignore_warning(category=RuntimeWarning, message='divide by zero encountered')
def testBinaryUfuncs(self, scalar_type, ufunc):
    x = np.array(VALUES[scalar_type])
    y = np.array(VALUES[scalar_type], dtype=scalar_type)
    np.testing.assert_array_equal(ufunc(x[:, None], x[None, :]).astype(scalar_type), ufunc(y[:, None], y[None, :]))