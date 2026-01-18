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
def testFromStr(self, float_type):
    self.assertEqual(float_type(1.2), float_type('1.2'))
    self.assertTrue(np.isnan(float_type('nan')))
    self.assertTrue(np.isnan(float_type('-nan')))
    if dtype_has_inf(float_type):
        self.assertEqual(float_type(float('inf')), float_type('inf'))
        self.assertEqual(float_type(float('-inf')), float_type('-inf'))