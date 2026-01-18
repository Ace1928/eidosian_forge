import numpy as np
import pandas as pd
import os
import pytest
from numpy.testing import assert_, assert_equal, assert_allclose
from statsmodels.tsa.statespace.exponential_smoothing import (
def test_estimated_params(self):
    res = self.mod.fit(self.start_params, disp=0, maxiter=100)
    np.set_printoptions(suppress=True)
    conc_res = self.conc_mod.fit(self.start_params[:len(self.params)], disp=0)
    assert_allclose(conc_res.llf, res.llf, atol=self.atol, rtol=self.rtol)
    assert_allclose(conc_res.initial_state, res.initial_state, atol=self.atol, rtol=self.rtol)