import warnings
import io
import numpy as np
from numpy.testing import (
from pytest import raises as assert_raises
import pytest
from scipy.interpolate import (
def test_nag(self):
    dataStr = '\n          7.99   0.00000E+0\n          8.09   0.27643E-4\n          8.19   0.43750E-1\n          8.70   0.16918E+0\n          9.20   0.46943E+0\n         10.00   0.94374E+0\n         12.00   0.99864E+0\n         15.00   0.99992E+0\n         20.00   0.99999E+0\n        '
    data = np.loadtxt(io.StringIO(dataStr))
    pch = pchip(data[:, 0], data[:, 1])
    resultStr = '\n           7.9900       0.0000\n           9.1910       0.4640\n          10.3920       0.9645\n          11.5930       0.9965\n          12.7940       0.9992\n          13.9950       0.9998\n          15.1960       0.9999\n          16.3970       1.0000\n          17.5980       1.0000\n          18.7990       1.0000\n          20.0000       1.0000\n        '
    result = np.loadtxt(io.StringIO(resultStr))
    assert_allclose(result[:, 1], pch(result[:, 0]), rtol=0.0, atol=5e-05)