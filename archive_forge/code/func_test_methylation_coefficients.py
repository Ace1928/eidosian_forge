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
def test_methylation_coefficients(self):
    rslt = self.meth_fit
    assert_close(rslt.params[:-2], expected_methylation_mean['Estimate'], 0.01)
    assert_close(rslt.tvalues[:-2], expected_methylation_mean['zvalue'], 0.1)
    assert_close(rslt.pvalues[:-2], expected_methylation_mean['Pr(>|z|)'], 0.01)