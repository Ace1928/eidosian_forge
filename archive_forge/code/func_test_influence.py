import io
import os
import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_equal
import pandas as pd
import patsy
from statsmodels.api import families
from statsmodels.tools.sm_exceptions import (
from statsmodels.othermod.betareg import BetaModel
from .results import results_betareg as resultsb
def test_influence(self):
    res1 = self.res1
    from statsmodels.stats.outliers_influence import MLEInfluence
    influ0 = MLEInfluence(res1)
    influ = res1.get_influence()
    attrs = ['cooks_distance', 'd_fittedvalues', 'd_fittedvalues_scaled', 'd_params', 'dfbetas', 'hat_matrix_diag', 'resid_studentized']
    for attr in attrs:
        getattr(influ, attr)
    frame = influ.summary_frame()
    frame0 = influ0.summary_frame()
    assert_allclose(frame, frame0, rtol=1e-13, atol=1e-13)