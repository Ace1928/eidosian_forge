import numpy as np
from numpy.testing import assert_allclose, assert_array_less
from scipy import stats
from statsmodels.distributions.copula.api import (
from statsmodels.distributions.copula.api import transforms as tra
import statsmodels.distributions.tools as dt
from statsmodels.distributions.bernstein import (
def test_bernstein_distribution_2d():
    grid = dt._Grid([51, 51])
    cop_tr = tra.TransfFrank
    args = (2,)
    ca = ArchimedeanCopula(cop_tr())
    distr1 = stats.uniform
    distr2 = stats.uniform
    cad = CopulaDistribution(ca, [distr1, distr2], cop_args=args)
    cdfv = cad.cdf(grid.x_flat, args)
    cdf_g = cdfv.reshape(grid.k_grid)
    bpd = BernsteinDistribution(cdf_g)
    cdf_bp = bpd.cdf(grid.x_flat)
    assert_allclose(cdf_bp, cdfv, atol=0.005)
    assert_array_less(np.median(np.abs(cdf_bp - cdfv)), 0.001)
    grid_eps = dt._Grid([51, 51], eps=1e-08)
    pdfv = cad.pdf(grid_eps.x_flat)
    pdf_bp = bpd.pdf(grid_eps.x_flat)
    assert_allclose(pdf_bp, pdfv, atol=0.01, rtol=0.04)
    assert_array_less(np.median(np.abs(pdf_bp - pdfv)), 0.05)
    xx = np.column_stack((np.linspace(0, 1, 5), np.ones(5)))
    cdf_m1 = bpd.cdf(xx)
    assert_allclose(cdf_m1, xx[:, 0], atol=1e-13)
    xx = np.column_stack((np.ones(5), np.linspace(0, 1, 5)))
    cdf_m2 = bpd.cdf(xx)
    assert_allclose(cdf_m2, xx[:, 1], atol=1e-13)
    xx_ = np.linspace(0, 1, 5)
    xx = xx_[:, None]
    bpd_m1 = bpd.get_marginal(0)
    cdf_m1 = bpd_m1.cdf(xx)
    assert_allclose(cdf_m1, xx_, atol=1e-13)
    pdf_m1 = bpd_m1.pdf(xx)
    assert_allclose(pdf_m1, np.ones(len(xx)), atol=1e-13)
    bpd_m = bpd.get_marginal(1)
    cdf_m = bpd_m.cdf(xx)
    assert_allclose(cdf_m, xx_, atol=1e-13)
    pdf_m = bpd_m.pdf(xx)
    assert_allclose(pdf_m, np.ones(len(xx)), atol=1e-13)