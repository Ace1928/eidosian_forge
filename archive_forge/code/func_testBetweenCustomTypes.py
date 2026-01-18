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
def testBetweenCustomTypes(self, float_type):
    for dtype in FLOAT_DTYPES:
        x = np.array(FLOAT_VALUES[float_type], dtype=dtype)
        y = x.astype(float_type)
        z = x.astype(float).astype(float_type)
        numpy_assert_allclose(y, z, float_type=float_type)