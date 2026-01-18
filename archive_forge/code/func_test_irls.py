from ..regression import least_squares, irls
from ..testing import requires
@requires('numpy')
def test_irls():
    import numpy as np
    x = np.linspace(0, 100)
    y = np.exp(-x / 47)
    b, c, info = irls(x, np.log(y))
    assert abs(b[1] + 1 / 47) < 1e-05
    assert np.all(c < 0.0001)
    assert info['success'] is True
    assert info['niter'] < 3