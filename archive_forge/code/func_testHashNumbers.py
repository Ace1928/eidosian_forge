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
def testHashNumbers(self, float_type):
    for value in np.extract(np.isfinite(FLOAT_VALUES[float_type]), FLOAT_VALUES[float_type]):
        with self.subTest(value):
            self.assertEqual(hash(value), hash(float_type(value)), str(value))