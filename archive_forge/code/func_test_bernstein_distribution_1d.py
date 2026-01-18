import numpy as np
from numpy.testing import assert_allclose, assert_array_less
from scipy import stats
from statsmodels.distributions.copula.api import (
from statsmodels.distributions.copula.api import transforms as tra
import statsmodels.distributions.tools as dt
from statsmodels.distributions.bernstein import (
def test_bernstein_distribution_1d():
    grid = dt._Grid([501])
    loc = grid.x_flat == 0
    grid.x_flat[loc] = grid.x_flat[~loc].min() / 2
    grid.x_flat[grid.x_flat == 1] = 1 - grid.x_flat.min()
    distr = stats.beta(3, 5)
    cdf_g = distr.cdf(np.squeeze(grid.x_flat))
    bpd = BernsteinDistribution(cdf_g)
    cdf_bp = bpd.cdf(grid.x_flat)
    assert_allclose(cdf_bp, cdf_g, atol=0.005)
    assert_array_less(np.median(np.abs(cdf_bp - cdf_g)), 0.001)
    pdfv = distr.pdf(np.squeeze(grid.x_flat))
    pdf_bp = bpd.pdf(grid.x_flat)
    assert_allclose(pdf_bp, pdfv, atol=0.02)
    assert_array_less(np.median(np.abs(pdf_bp - pdfv)), 0.01)
    xf = np.squeeze(grid.x_flat)
    bpd1 = BernsteinDistributionUV(cdf_g)
    cdf_bp1 = bpd1.cdf(xf)
    assert_allclose(cdf_bp1, cdf_bp, atol=1e-13)
    pdf_bp1 = bpd1.pdf(xf)
    assert_allclose(pdf_bp1, pdf_bp, atol=1e-13)
    cdf_bp1 = bpd1.cdf(xf, method='beta')
    assert_allclose(cdf_bp1, cdf_bp, atol=1e-13)
    pdf_bp1 = bpd1.pdf(xf, method='beta')
    assert_allclose(pdf_bp1, pdf_bp, atol=1e-13)
    cdf_bp1 = bpd1.cdf(xf, method='bpoly')
    assert_allclose(cdf_bp1, cdf_bp, atol=1e-13)
    pdf_bp1 = bpd1.pdf(xf, method='bpoly')
    assert_allclose(pdf_bp1, pdf_bp, atol=1e-13)
    rvs = bpd.rvs(100)
    assert len(rvs) == 100