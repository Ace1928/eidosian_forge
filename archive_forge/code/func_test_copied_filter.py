import copy
import pickle
import numpy as np
import pandas as pd
import os
import pytest
from scipy.linalg.blas import find_best_blas_type
from scipy.linalg import solve_discrete_lyapunov
from statsmodels.tsa.statespace.kalman_filter import (
from statsmodels.tsa.statespace.mlemodel import MLEModel
from statsmodels.tsa.statespace import _representation, _kalman_filter
from .results import results_kalman_filter
from numpy.testing import assert_almost_equal, assert_allclose
def test_copied_filter(self):
    copied = copy.deepcopy(self.filter)
    self.filter()
    copied()
    assert id(filter) != id(copied)
    assert_allclose(np.array(self.filter.filtered_state), np.array(copied.filtered_state))
    assert_allclose(np.array(self.filter.loglikelihood), np.array(copied.loglikelihood))