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
def testEqual2(self, float_type):
    a = np.array([31], float_type)
    b = np.array([15], float_type)
    self.assertFalse(a.__eq__(b))