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
def testArgmax(self, float_type):
    values_to_sort = np.float32(float_type(np.float32(FLOAT_VALUES[float_type])))
    argmax_f32 = np.argmax(values_to_sort)
    argmax_float_type = np.argmax(values_to_sort.astype(float_type))
    np.testing.assert_equal(argmax_f32, argmax_float_type)