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
def testSub(self, float_type):
    for a, b in [(0, 0), (1, 0), (1, -1), (2, 3.5), (3.5, -2.25), (-2.25, float('inf')), (-2.25, float('-inf')), (3.5, float('nan'))]:
        binary_operation_test(a, b, op=lambda a, b: a - b, float_type=float_type)