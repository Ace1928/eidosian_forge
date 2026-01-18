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
def testCopySign(self, float_type):
    for bits in list(range(1, 128)):
        with self.subTest(bits):
            bits_type = BITS_TYPE[float_type]
            val = bits_type(bits).view(float_type)
            val_with_sign = np.copysign(val, float_type(-1))
            val_with_sign_bits = val_with_sign.view(bits_type)
            num_bits = np.iinfo(bits_type).bits
            np.testing.assert_equal(bits | 1 << num_bits - 1, val_with_sign_bits)