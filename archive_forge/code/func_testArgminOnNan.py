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
def testArgminOnNan(self, float_type):
    """Ensures we return the right thing for multiple NaNs."""
    one_with_nans = np.array([1.0, float('nan'), float('nan')], dtype=np.float32)
    np.testing.assert_equal(np.argmin(one_with_nans.astype(float_type)), np.argmin(one_with_nans))