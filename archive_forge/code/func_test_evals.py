import os
import warnings
import numpy as np
from numpy.testing import assert_almost_equal, assert_equal
import pandas as pd
import pytest
from statsmodels.tools.sm_exceptions import HypothesisTestWarning
from statsmodels.tsa.vector_ar.vecm import coint_johansen
def test_evals(self):
    assert_almost_equal(self.res.eig, self.eig_m)