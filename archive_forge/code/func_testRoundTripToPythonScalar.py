import contextlib
import copy
import operator
import pickle
import warnings
from absl.testing import absltest
from absl.testing import parameterized
import ml_dtypes
import numpy as np
@parameterized.product(scalar_type=INT4_TYPES, python_scalar=[int, float])
def testRoundTripToPythonScalar(self, scalar_type, python_scalar):
    for v in VALUES[scalar_type]:
        self.assertEqual(v, scalar_type(v))
        self.assertEqual(python_scalar(v), python_scalar(scalar_type(v)))
        self.assertEqual(scalar_type(v), scalar_type(python_scalar(scalar_type(v))))