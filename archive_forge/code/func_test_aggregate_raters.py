import numpy as np
from numpy.testing import assert_almost_equal, assert_equal, assert_allclose
from statsmodels.stats.inter_rater import (fleiss_kappa, cohens_kappa,
from statsmodels.tools.testing import Holder
def test_aggregate_raters():
    data = diagnoses
    data_, categories = aggregate_raters(data)
    colsum = np.array([26, 26, 30, 55, 43])
    assert_equal(data_.sum(0), colsum)
    assert_equal(np.unique(diagnoses), categories)