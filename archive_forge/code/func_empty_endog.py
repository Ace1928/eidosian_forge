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
def empty_endog():
    endog = np.zeros((0, 0))
    Representation(endog, k_states=2)