import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_almost_equal, assert_equal
from skimage._shared.testing import fetch
from skimage._shared.utils import _supported_float_type
from skimage.color.delta_e import (
@pytest.mark.parametrize('channel_axis', [0, 1, -1])
@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_cie76(dtype, channel_axis):
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
    dE2 = deltaE_cie76(lab1, lab2, channel_axis=channel_axis)
    assert dE2.dtype == _supported_float_type(dtype)
    oracle = np.array([4.00106328, 6.31415011, 9.1776999, 2.06270077, 2.36957073, 2.91529271, 2.23606798, 2.23606798, 4.98000036, 4.9800004, 4.98000044, 4.98000049, 4.98000036, 4.9800004, 4.98000044, 3.53553391, 36.86800781, 31.91002977, 30.25309901, 27.40894015, 0.89242934, 0.7972, 0.8583065, 0.82982507, 3.1819238, 2.21334297, 1.53890382, 4.60630929, 6.58467989, 3.88641412, 1.50514845, 2.3237848, 0.94413208, 1.31910843])
    rtol = 1e-05 if dtype == np.float32 else 1e-08
    assert_allclose(dE2, oracle, rtol=rtol)