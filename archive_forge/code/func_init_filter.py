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
@classmethod
def init_filter(cls):
    prefix = find_best_blas_type((cls.obs,))
    klass = prefix_statespace_map[prefix[0]]
    model = klass(cls.obs, cls.design, cls.obs_intercept, cls.obs_cov, cls.transition, cls.state_intercept, cls.selection, cls.state_cov)
    model.initialize_known(cls.initial_state, cls.initial_state_cov)
    klass = prefix_kalman_filter_map[prefix[0]]
    kfilter = klass(model, conserve_memory=cls.conserve_memory, loglikelihood_burn=cls.loglikelihood_burn)
    return (model, kfilter)