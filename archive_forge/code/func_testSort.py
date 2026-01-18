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
def testSort(self, float_type):
    values_to_sort = np.float32([x for x in FLOAT_VALUES[float_type] if not np.isnan(x)])
    sorted_f32 = np.sort(values_to_sort)
    sorted_float_type = np.sort(values_to_sort.astype(float_type))
    np.testing.assert_equal(sorted_f32, np.float32(sorted_float_type))