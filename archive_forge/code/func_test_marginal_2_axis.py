from scipy import stats, linalg, integrate
import numpy as np
from numpy.testing import (assert_almost_equal, assert_, assert_equal,
import pytest
from pytest import raises as assert_raises
@pytest.mark.xslow
def test_marginal_2_axis():
    rng = np.random.default_rng(6111799263660870475)
    n_data = 30
    n_dim = 4
    dataset = rng.normal(size=(n_dim, n_data))
    points = rng.normal(size=(n_dim, 3))
    dimensions = np.array([1, 3])
    kde = stats.gaussian_kde(dataset)
    marginal = kde.marginal(dimensions)
    pdf = marginal.pdf(points[dimensions])

    def marginal_pdf(points):

        def marginal_pdf_single(point):

            def f(y, x):
                w, z = point[dimensions]
                x = np.array([x, w, y, z])
                return kde.pdf(x)[0]
            return integrate.dblquad(f, -np.inf, np.inf, -np.inf, np.inf)[0]
        return np.apply_along_axis(marginal_pdf_single, axis=0, arr=points)
    ref = marginal_pdf(points)
    assert_allclose(pdf, ref, rtol=1e-06)