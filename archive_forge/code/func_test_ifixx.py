import tempfile
import shutil
import os
import numpy as np
from numpy import pi
from numpy.testing import (assert_array_almost_equal,
import pytest
from pytest import raises as assert_raises
from scipy.odr import (Data, Model, ODR, RealData, OdrStop, OdrWarning,
def test_ifixx(self):
    x1 = [-2.01, -0.99, -0.001, 1.02, 1.98]
    x2 = [3.98, 1.01, 0.001, 0.998, 4.01]
    fix = np.vstack((np.zeros_like(x1, dtype=int), np.ones_like(x2, dtype=int)))
    data = Data(np.vstack((x1, x2)), y=1, fix=fix)
    model = Model(lambda beta, x: x[1, :] - beta[0] * x[0, :] ** 2.0, implicit=True)
    odr1 = ODR(data, model, beta0=np.array([1.0]))
    sol1 = odr1.run()
    odr2 = ODR(data, model, beta0=np.array([1.0]), ifixx=fix)
    sol2 = odr2.run()
    assert_equal(sol1.beta, sol2.beta)