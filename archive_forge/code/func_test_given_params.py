import numpy as np
import pandas as pd
import os
import pytest
from numpy.testing import assert_, assert_equal, assert_allclose
from statsmodels.tsa.statespace.exponential_smoothing import (
def test_given_params(self):
    res = self.mod.fit_constrained(self.params.to_dict(), disp=0)
    conc_res = self.conc_mod.filter(self.params.values)
    assert_allclose(conc_res.llf, res.llf, atol=self.atol, rtol=self.rtol)
    assert_allclose(conc_res.initial_state, res.initial_state, atol=self.atol, rtol=self.rtol)