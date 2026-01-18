import tempfile
import shutil
import os
import numpy as np
from numpy import pi
from numpy.testing import (assert_array_almost_equal,
import pytest
from pytest import raises as assert_raises
from scipy.odr import (Data, Model, ODR, RealData, OdrStop, OdrWarning,
def lorentz(self, beta, x):
    return beta[0] * beta[1] * beta[2] / np.sqrt(np.power(x * x - beta[2] * beta[2], 2.0) + np.power(beta[1] * x, 2.0))