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
@ignore_warning(category=RuntimeWarning, message='overflow encountered')
def testRoundTripNumpyTypes(self, float_type):
    for dtype in [np.float16, np.float32, np.float64, np.longdouble]:
        for f in FLOAT_VALUES[float_type]:
            np.testing.assert_equal(dtype(f), dtype(float_type(dtype(f))))
            np.testing.assert_equal(float(dtype(f)), float(float_type(dtype(f))))
            np.testing.assert_equal(dtype(f), dtype(float_type(np.array(f, dtype))))
        np.testing.assert_equal(dtype(np.array(FLOAT_VALUES[float_type], float_type)), np.array(FLOAT_VALUES[float_type], dtype))