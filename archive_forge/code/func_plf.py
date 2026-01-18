import itertools
import os
import numpy as np
from statsmodels.duration.hazard_regression import PHReg
from numpy.testing import (assert_allclose,
import pandas as pd
import pytest
from .results import survival_r_results
from .results import survival_enet_r_results
def plf(params):
    llf = model.loglike(params) / len(time)
    L1_wt = 1
    llf = llf - s * ((1 - L1_wt) * np.sum(params ** 2) / 2 + L1_wt * np.sum(np.abs(params)))
    return llf