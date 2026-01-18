import numpy as np
from numpy.testing import assert_allclose, assert_equal
from scipy import stats
import statsmodels.distributions.tools as dt
def test_grid_class():
    res = {'k_grid': [3, 5], 'x_marginal': [np.array([0.0, 0.5, 1.0]), np.array([0.0, 0.25, 0.5, 0.75, 1.0])], 'idx_flat.T': np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0], [0.0, 1.0, 2.0, 3.0, 4.0, 0.0, 1.0, 2.0, 3.0, 4.0, 0.0, 1.0, 2.0, 3.0, 4.0]])}
    gg = dt._Grid([3, 5])
    assert_equal(gg.k_grid, res['k_grid'])
    assert gg.x_marginal, res['x_marginal']
    assert_allclose(gg.idx_flat, res['idx_flat.T'].T, atol=1e-12)
    assert_allclose(gg.x_flat, res['idx_flat.T'].T / [2, 4], atol=1e-12)
    gg = dt._Grid([3, 5], eps=0.001)
    assert_allclose(gg.x_flat.min(), 0.001, atol=1e-12)
    assert_allclose(gg.x_flat.max(), 0.999, atol=1e-12)
    xmf = np.concatenate(gg.x_marginal)
    assert_allclose(xmf.min(), 0.001, atol=1e-12)
    assert_allclose(xmf.max(), 0.999, atol=1e-12)
    gg = dt._Grid([5], eps=0.001)
    res = {'k_grid': [5], 'x_marginal': [np.array([0.001, 0.25, 0.5, 0.75, 0.999])], 'idx_flat.T': np.array([[0.0, 1.0, 2.0, 3.0, 4.0]])}
    assert_equal(gg.k_grid, res['k_grid'])
    assert gg.x_marginal, res['x_marginal']
    assert_allclose(gg.idx_flat, res['idx_flat.T'].T, atol=1e-12)
    assert_allclose(gg.x_flat, res['x_marginal'][0][:, None], atol=1e-12)
    gg = dt._Grid([3, 3, 2], eps=0.0)
    res = {'k_grid': [3, 3, 2], 'x_marginal': [np.array([0.0, 0.5, 1.0]), np.array([0.0, 0.5, 1.0]), np.array([0.0, 1.0])], 'idx_flat.T': np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0], [0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 0.0, 0.0, 1.0, 1.0, 2.0, 2.0], [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0]])}
    assert_equal(gg.k_grid, res['k_grid'])
    assert gg.x_marginal, res['x_marginal']
    assert_allclose(gg.idx_flat, res['idx_flat.T'].T, atol=1e-12)
    assert_allclose(gg.x_flat, res['idx_flat.T'].T / [2, 2, 1], atol=1e-12)