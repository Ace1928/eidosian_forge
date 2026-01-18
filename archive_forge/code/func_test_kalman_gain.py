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
def test_kalman_gain(self):
    assert_allclose(self.results.kalman_gain.sum(axis=1).sum(axis=0), clark1989_results['V1'], atol=0.0001)