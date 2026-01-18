import tempfile
import shutil
import os
import numpy as np
from numpy import pi
from numpy.testing import (assert_array_almost_equal,
import pytest
from pytest import raises as assert_raises
from scipy.odr import (Data, Model, ODR, RealData, OdrStop, OdrWarning,
def df_dbeta_odr(beta, x):
    nr_meas = np.shape(x)[1]
    zeros = np.zeros(nr_meas)
    ones = np.ones(nr_meas)
    dy0 = np.array([ones, x[0, :], x[1, :], zeros, zeros, zeros])
    dy1 = np.array([zeros, zeros, zeros, ones, x[0, :], x[1, :]])
    return np.stack((dy0, dy1))