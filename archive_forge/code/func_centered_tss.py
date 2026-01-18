from __future__ import annotations
from statsmodels.compat.pandas import Appender
from statsmodels.compat.python import lrange, lzip
from typing import Literal
from collections.abc import Sequence
import warnings
import numpy as np
from scipy import optimize, stats
from scipy.linalg import cholesky, toeplitz
from scipy.linalg.lapack import dtrtri
import statsmodels.base.model as base
import statsmodels.base.wrapper as wrap
from statsmodels.emplike.elregress import _ELRegOpts
from statsmodels.regression._prediction import PredictionResults
from statsmodels.tools.decorators import cache_readonly, cache_writable
from statsmodels.tools.sm_exceptions import (
from statsmodels.tools.tools import pinv_extended
from statsmodels.tools.typing import Float64Array
from statsmodels.tools.validation import bool_like, float_like, string_like
from . import _prediction as pred
@cache_readonly
def centered_tss(self):
    """The total (weighted) sum of squares centered about the mean."""
    model = self.model
    weights = getattr(model, 'weights', None)
    sigma = getattr(model, 'sigma', None)
    if weights is not None:
        mean = np.average(model.endog, weights=weights)
        return np.sum(weights * (model.endog - mean) ** 2)
    elif sigma is not None:
        iota = np.ones_like(model.endog)
        iota = model.whiten(iota)
        mean = model.wendog.dot(iota) / iota.dot(iota)
        err = model.endog - mean
        err = model.whiten(err)
        return np.sum(err ** 2)
    else:
        centered_endog = model.wendog - model.wendog.mean()
        return np.dot(centered_endog, centered_endog)