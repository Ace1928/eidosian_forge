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
@ignore_warning(category=RuntimeWarning, message='invalid value encountered')
@ignore_warning(category=RuntimeWarning, message='divide by zero encountered')
def testUnaryUfunc(self, float_type):
    for op in UNARY_UFUNCS:
        with self.subTest(op.__name__):
            rng = np.random.RandomState(seed=42)
            x = rng.randn(3, 7, 10).astype(float_type)
            numpy_assert_allclose(op(x).astype(np.float32), truncate(op(x.astype(np.float32)), float_type=float_type), rtol=0.0001, float_type=float_type)