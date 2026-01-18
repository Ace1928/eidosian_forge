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
def testRoundTripToFloat(self, float_type):
    for v in FLOAT_VALUES[float_type]:
        np.testing.assert_equal(v, float(float_type(v)))