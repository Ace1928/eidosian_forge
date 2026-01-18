import tempfile
import shutil
import os
import numpy as np
from numpy import pi
from numpy.testing import (assert_array_almost_equal,
import pytest
from pytest import raises as assert_raises
from scipy.odr import (Data, Model, ODR, RealData, OdrStop, OdrWarning,
def implicit_fcn(self, B, x):
    return B[2] * np.power(x[0] - B[0], 2) + 2.0 * B[3] * (x[0] - B[0]) * (x[1] - B[1]) + B[4] * np.power(x[1] - B[1], 2) - 1.0