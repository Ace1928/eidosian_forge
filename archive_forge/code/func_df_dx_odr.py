import tempfile
import shutil
import os
import numpy as np
from numpy import pi
from numpy.testing import (assert_array_almost_equal,
import pytest
from pytest import raises as assert_raises
from scipy.odr import (Data, Model, ODR, RealData, OdrStop, OdrWarning,
def df_dx_odr(beta, x):
    nr_meas = np.shape(x)[1]
    ones = np.ones(nr_meas)
    dy0 = np.array([beta[1] * ones, beta[2] * ones])
    dy1 = np.array([beta[4] * ones, beta[5] * ones])
    return np.stack((dy0, dy1))