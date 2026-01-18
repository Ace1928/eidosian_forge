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
def testFrexp(self, float_type):
    rng = np.random.RandomState(seed=42)
    x = rng.randn(3, 7).astype(float_type)
    mant1, exp1 = np.frexp(x)
    mant2, exp2 = np.frexp(x.astype(np.float32))
    np.testing.assert_equal(exp1, exp2)
    numpy_assert_allclose(mant1, mant2, rtol=0.01, float_type=float_type)