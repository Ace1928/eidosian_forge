import numpy as np
from numpy.testing import assert_almost_equal, assert_equal, assert_allclose
from statsmodels.stats.inter_rater import (fleiss_kappa, cohens_kappa,
from statsmodels.tools.testing import Holder
def test_fleiss_kappa_irr():
    fleiss = Holder()
    fleiss.method = "Fleiss' Kappa for m Raters"
    fleiss.irr_name = 'Kappa'
    fleiss.value = 0.4302445
    fleiss.stat_name = 'z'
    fleiss.statistic = 17.65183
    fleiss.p_value = 0
    data_, _ = aggregate_raters(diagnoses)
    res1_kappa = fleiss_kappa(data_)
    assert_almost_equal(res1_kappa, fleiss.value, decimal=7)