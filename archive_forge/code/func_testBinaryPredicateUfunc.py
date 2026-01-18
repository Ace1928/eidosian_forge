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
def testBinaryPredicateUfunc(self, float_type):
    for op in BINARY_PREDICATE_UFUNCS:
        with self.subTest(op.__name__):
            rng = np.random.RandomState(seed=42)
            x = rng.randn(3, 7).astype(float_type)
            y = rng.randn(4, 1, 7).astype(float_type)
            np.testing.assert_equal(op(x, y), op(x.astype(np.float32), y.astype(np.float32)))