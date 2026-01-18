from statsmodels.compat.platform import PLATFORM_OSX
import os
import csv
import warnings
import numpy as np
import pandas as pd
from scipy import sparse
import pytest
from statsmodels.regression.mixed_linear_model import (
from numpy.testing import (assert_almost_equal, assert_equal, assert_allclose,
from statsmodels.base import _penalties as penalties
import statsmodels.tools.numdiff as nd
from .results import lme_r_results
def loglike_function(model, profile_fe, has_fe):

    def f(x):
        params = MixedLMParams.from_packed(x, model.k_fe, model.k_re, model.use_sqrt, has_fe=has_fe)
        return -model.loglike(params, profile_fe=profile_fe)
    return f