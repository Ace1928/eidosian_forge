import warnings
import numpy as np
from numpy.testing import assert_allclose, assert_equal
import pytest
from statsmodels.tools.tools import add_constant
from statsmodels.base._prediction_inference import PredictionResultsMonotonic
from statsmodels.discrete.discrete_model import (
from statsmodels.discrete.count_model import (
from statsmodels.sandbox.regression.tests.test_gmm_poisson import DATA
from .results import results_predict as resp
def test_diagnostic(self):
    res1 = self.res1
    dia = res1.get_diagnostic(y_max=21)
    res_chi2 = dia.test_chisquare_prob(bin_edges=np.arange(4))
    assert_equal(res_chi2.diff1.shape[1], 3)
    assert_equal(dia.probs_predicted.shape[1], 22)
    try:
        dia.plot_probs(upp_xlim=20)
    except ImportError:
        pass