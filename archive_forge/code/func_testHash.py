import contextlib
import copy
import operator
import pickle
import warnings
from absl.testing import absltest
from absl.testing import parameterized
import ml_dtypes
import numpy as np
@parameterized.product(scalar_type=INT4_TYPES)
def testHash(self, scalar_type):
    for v in VALUES[scalar_type]:
        self.assertEqual(hash(v), hash(scalar_type(v)), msg=v)