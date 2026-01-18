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
def testPredicateUfunc(self, float_type):
    for op in [np.isfinite, np.isinf, np.isnan, np.signbit, np.logical_not]:
        with self.subTest(op.__name__):
            rng = np.random.RandomState(seed=42)
            shape = (3, 7, 10)
            posinf_flips = rng.rand(*shape) < 0.1
            neginf_flips = rng.rand(*shape) < 0.1
            nan_flips = rng.rand(*shape) < 0.1
            vals = rng.randn(*shape)
            vals = np.where(posinf_flips, np.inf, vals)
            vals = np.where(neginf_flips, -np.inf, vals)
            vals = np.where(nan_flips, np.nan, vals)
            vals = vals.astype(float_type)
            np.testing.assert_equal(op(vals), op(vals.astype(np.float32)))