import numpy as np
from numpy.testing import assert_almost_equal
import numpy.testing as npt
import statsmodels.tools.eval_measures as em
from statsmodels.stats.moment_helpers import cov2corr, se_cov
from statsmodels.regression.linear_model import OLS
from statsmodels.sandbox.panel.panel_short import ShortPanelGLS, ShortPanelGLS2
from statsmodels.sandbox.panel.random_panel import PanelSample
import statsmodels.sandbox.panel.correlation_structures as cs
import statsmodels.stats.sandwich_covariance as sw
def assert_maxabs(actual, expected, value):
    npt.assert_array_less(em.maxabs(actual, expected, None), value)