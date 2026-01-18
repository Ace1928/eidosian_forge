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
def testModuleName(self, float_type):
    self.assertEqual(float_type.__module__, 'ml_dtypes')