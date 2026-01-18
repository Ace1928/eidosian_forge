from scipy import stats, linalg, integrate
import numpy as np
from numpy.testing import (assert_almost_equal, assert_, assert_equal,
import pytest
from pytest import raises as assert_raises
def marginal_pdf(points):

    def marginal_pdf_single(point):

        def f(y, x):
            w, z = point[dimensions]
            x = np.array([x, w, y, z])
            return kde.pdf(x)[0]
        return integrate.dblquad(f, -np.inf, np.inf, -np.inf, np.inf)[0]
    return np.apply_along_axis(marginal_pdf_single, axis=0, arr=points)