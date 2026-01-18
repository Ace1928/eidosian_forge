import os
import warnings
import numpy as np
import pandas as pd
import pytest
from statsmodels.tsa.statespace.representation import Representation
from statsmodels.tsa.statespace.kalman_filter import (
from statsmodels.tsa.statespace.simulation_smoother import SimulationSmoother
from statsmodels.tsa.statespace import tools, sarimax
from .results import results_kalman_filter
from numpy.testing import (
def test_no_endog():
    mod = KalmanFilter(k_endog=1, k_states=1)
    assert_raises(RuntimeError, mod._initialize_filter)
    mod.initialize_approximate_diffuse()
    assert_raises(RuntimeError, mod.filter)