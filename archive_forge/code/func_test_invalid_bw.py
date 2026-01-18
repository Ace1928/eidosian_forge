import pytest
import numpy as np
import numpy.testing as npt
import statsmodels.api as sm
def test_invalid_bw():
    x = np.arange(400)
    y = x ** 2
    with pytest.raises(ValueError):
        nparam.KernelReg(x, y, 'c', bw=[12.5, 1.0])