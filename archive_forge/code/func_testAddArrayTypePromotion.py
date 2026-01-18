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
def testAddArrayTypePromotion(self, float_type):
    self.assertEqual(np.float32, type(float_type(3.5) + np.array(2.25, np.float32)))
    self.assertEqual(np.float32, type(np.array(3.5, np.float32) + float_type(2.25)))