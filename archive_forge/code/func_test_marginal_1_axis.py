from scipy import stats, linalg, integrate
import numpy as np
from numpy.testing import (assert_almost_equal, assert_, assert_equal,
import pytest
from pytest import raises as assert_raises
def test_marginal_1_axis():
    rng = np.random.default_rng(6111799263660870475)
    n_data = 50
    n_dim = 10
    dataset = rng.normal(size=(n_dim, n_data))
    points = rng.normal(size=(n_dim, 3))
    dimensions = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
    kde = stats.gaussian_kde(dataset)
    marginal = kde.marginal(dimensions)
    pdf = marginal.pdf(points[dimensions])

    def marginal_pdf_single(point):

        def f(x):
            x = np.concatenate(([x], point[dimensions]))
            return kde.pdf(x)[0]
        return integrate.quad(f, -np.inf, np.inf)[0]

    def marginal_pdf(points):
        return np.apply_along_axis(marginal_pdf_single, axis=0, arr=points)
    ref = marginal_pdf(points)
    assert_allclose(pdf, ref, rtol=1e-06)