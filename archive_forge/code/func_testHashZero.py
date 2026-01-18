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
def testHashZero(self, float_type):
    """Tests that negative zero and zero hash to the same value."""
    self.assertEqual(hash(float_type(-0.0)), hash(float_type(0.0)))