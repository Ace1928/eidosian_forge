import numpy as np
from statsmodels.tsa.interp import dentonm
def test_denton_quarterly2():
    zQ = np.array([50, 100, 150, 100] * 5)
    Y = np.array([500, 400, 300, 400, 500])
    x_denton = dentonm(zQ, Y, freq='aq')
    x_stata = np.array([64.334796, 127.80616, 187.82379, 120.03526, 56.563894, 105.97568, 147.50144, 89.958987, 40.547201, 74.445963, 108.34473, 76.66211, 42.763347, 94.14664, 153.41596, 109.67405, 58.290761, 122.62556, 190.41409, 128.66959])
    np.testing.assert_almost_equal(x_denton, x_stata, 5)