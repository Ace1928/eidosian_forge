import warnings
import os
import numpy as np
from numpy.testing import assert_equal, assert_allclose
import pandas as pd
import pytest
from statsmodels.tools import add_constant
from statsmodels.tsa.regime_switching import markov_autoregression
def test_filtered_regimes(self):
    assert_allclose(self.result.filtered_marginal_probabilities[0], self.mar_filardo['filtered_0'].iloc[5:], atol=1e-05)