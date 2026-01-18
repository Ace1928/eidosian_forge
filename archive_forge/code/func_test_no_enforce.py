import os
import re
import warnings
import numpy as np
from numpy.testing import assert_equal, assert_raises, assert_allclose
import pandas as pd
import pytest
from statsmodels.tsa.statespace import dynamic_factor
from .results import results_varmax, results_dynamic_factor
from statsmodels.iolib.summary import forg
def test_no_enforce(self):
    return
    params = self.model.untransform_params(self.true['params'])
    params[self.model._params_transition] = self.true['params'][self.model._params_transition]
    self.model.enforce_stationarity = False
    results = self.model.filter(params, transformed=False)
    self.model.enforce_stationarity = True
    assert_allclose(results.llf, self.results.llf, rtol=1e-05)