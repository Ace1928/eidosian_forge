from os.path import join, dirname
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_equal
import pytest
from pytest import raises as assert_raises
from scipy.fftpack._realtransforms import (
def idst_2d_ref(x, **kwargs):
    """Calculate reference values for testing idst2."""
    x = np.array(x, copy=True)
    for row in range(x.shape[0]):
        x[row, :] = idst(x[row, :], **kwargs)
    for col in range(x.shape[1]):
        x[:, col] = idst(x[:, col], **kwargs)
    return x