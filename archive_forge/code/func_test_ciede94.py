import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_almost_equal, assert_equal
from skimage._shared.testing import fetch
from skimage._shared.utils import _supported_float_type
from skimage.color.delta_e import (
@pytest.mark.parametrize('channel_axis', [0, 1, -1])
@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_ciede94(dtype, channel_axis):
    data = load_ciede2000_data()
    N = len(data)
    lab1 = np.zeros((N, 3), dtype=dtype)
    lab1[:, 0] = data['L1']
    lab1[:, 1] = data['a1']
    lab1[:, 2] = data['b1']
    lab2 = np.zeros((N, 3), dtype=dtype)
    lab2[:, 0] = data['L2']
    lab2[:, 1] = data['a2']
    lab2[:, 2] = data['b2']
    lab1 = np.moveaxis(lab1, source=-1, destination=channel_axis)
    lab2 = np.moveaxis(lab2, source=-1, destination=channel_axis)
    dE2 = deltaE_ciede94(lab1, lab2, channel_axis=channel_axis)
    assert dE2.dtype == _supported_float_type(dtype)
    oracle = np.array([1.39503887, 1.93410055, 2.45433566, 0.68449187, 0.6695627, 0.69194527, 2.23606798, 2.03163832, 4.80069441, 4.80069445, 4.80069449, 4.80069453, 4.80069441, 4.80069445, 4.80069449, 3.40774352, 34.6891632, 29.44137328, 27.91408781, 24.93766082, 0.82213163, 0.71658427, 0.8048753, 0.75284394, 1.39099471, 1.24808929, 1.29795787, 1.82045088, 2.55613309, 1.42491303, 1.41945261, 2.3225685, 0.93853308, 1.30654464])
    rtol = 1e-05 if dtype == np.float32 else 1e-08
    assert_allclose(dE2, oracle, rtol=rtol)