import numpy as np
from numpy.testing import assert_almost_equal, assert_equal, assert_allclose
from statsmodels.stats.inter_rater import (fleiss_kappa, cohens_kappa,
from statsmodels.tools.testing import Holder
def test_cohens_kappa_irr():
    ck_w3 = Holder()
    ck_w4 = Holder()
    ck_w3.method = "Cohen's Kappa for 2 Raters (Weights: 0,0,0,1,1,1)"
    ck_w3.irr_name = 'Kappa'
    ck_w3.value = 0.1891892
    ck_w3.stat_name = 'z'
    ck_w3.statistic = 0.5079002
    ck_w3.p_value = 0.6115233
    ck_w4.method = "Cohen's Kappa for 2 Raters (Weights: 0,0,1,1,2,2)"
    ck_w4.irr_name = 'Kappa'
    ck_w4.value = 0.2820513
    ck_w4.stat_name = 'z'
    ck_w4.statistic = 1.25741
    ck_w4.p_value = 0.2086053
    ck_w1 = Holder()
    ck_w2 = Holder()
    ck_w3 = Holder()
    ck_w4 = Holder()
    ck_w1.method = "Cohen's Kappa for 2 Raters (Weights: unweighted)"
    ck_w1.irr_name = 'Kappa'
    ck_w1.value = -0.006289308
    ck_w1.stat_name = 'z'
    ck_w1.statistic = -0.0604067
    ck_w1.p_value = 0.9518317
    ck_w2.method = "Cohen's Kappa for 2 Raters (Weights: equal)"
    ck_w2.irr_name = 'Kappa'
    ck_w2.value = 0.1459075
    ck_w2.stat_name = 'z'
    ck_w2.statistic = 1.282472
    ck_w2.p_value = 0.1996772
    ck_w3.method = "Cohen's Kappa for 2 Raters (Weights: squared)"
    ck_w3.irr_name = 'Kappa'
    ck_w3.value = 0.2520325
    ck_w3.stat_name = 'z'
    ck_w3.statistic = 1.437451
    ck_w3.p_value = 0.1505898
    ck_w4.method = "Cohen's Kappa for 2 Raters (Weights: 0,0,1,1,2)"
    ck_w4.irr_name = 'Kappa'
    ck_w4.value = 0.2391304
    ck_w4.stat_name = 'z'
    ck_w4.statistic = 1.223734
    ck_w4.p_value = 0.2210526
    all_cases = [(ck_w1, None, None), (ck_w2, None, 'linear'), (ck_w2, np.arange(5), None), (ck_w2, np.arange(5), 'toeplitz'), (ck_w3, None, 'quadratic'), (ck_w3, np.arange(5) ** 2, 'toeplitz'), (ck_w3, 4 * np.arange(5) ** 2, 'toeplitz'), (ck_w4, [0, 0, 1, 1, 2], 'toeplitz')]
    r = np.histogramdd(anxiety[:, 1:], ([1, 2, 3, 4, 6, 7], [1, 2, 3, 4, 6, 7]))
    for res2, w, wt in all_cases:
        msg = repr(w) + repr(wt)
        res1 = cohens_kappa(r[0], weights=w, wt=wt)
        assert_almost_equal(res1.kappa, res2.value, decimal=6, err_msg=msg)
        assert_almost_equal(res1.z_value, res2.statistic, decimal=5, err_msg=msg)
        assert_almost_equal(res1.pvalue_two_sided, res2.p_value, decimal=6, err_msg=msg)