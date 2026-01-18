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
@ignore_warning(category=RuntimeWarning, message='invalid value encountered')
def testSpacing(self, float_type):
    with self.subTest(name='Subnormals'):
        for i in range(int(np.log2(float(ml_dtypes.finfo(float_type).smallest_subnormal))), int(np.log2(float(ml_dtypes.finfo(float_type).smallest_normal)))):
            power_of_two = float_type(2.0 ** i)
            distance = ml_dtypes.finfo(float_type).smallest_subnormal
            np.testing.assert_equal(np.spacing(power_of_two), distance)
            np.testing.assert_equal(np.spacing(-power_of_two), -distance)
    with self.subTest(name='Normals'):
        for i in range(int(np.log2(float(ml_dtypes.finfo(float_type).smallest_normal))), int(np.log2(float(ml_dtypes.finfo(float_type).max)))):
            power_of_two = float_type(2.0 ** i)
            distance = ml_dtypes.finfo(float_type).eps * power_of_two
            np.testing.assert_equal(np.spacing(power_of_two), distance)
            np.testing.assert_equal(np.spacing(-power_of_two), -distance)
    with self.subTest(name='NextAfter'):
        for x in FLOAT_VALUES[float_type]:
            x_float_type = float_type(x)
            spacing = np.spacing(x_float_type)
            toward = np.copysign(float_type(2.0 * np.abs(x) + 1), x_float_type)
            nextup = np.nextafter(x_float_type, toward)
            if np.isnan(spacing):
                self.assertTrue(np.isnan(nextup - x_float_type))
            else:
                np.testing.assert_equal(spacing, nextup - x_float_type)
    with self.subTest(name='NonFinite'):
        nan = float_type(float('nan'))
        np.testing.assert_equal(np.spacing(nan), np.spacing(np.float32(nan)))
        if dtype_has_inf(float_type):
            inf = float_type(float('inf'))
            np.testing.assert_equal(np.spacing(inf), np.spacing(np.float32(inf)))