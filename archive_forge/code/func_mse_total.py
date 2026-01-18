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
def mse_total(self):
    """
        Total mean squared error.

        The uncentered total sum of squares divided by the number of
        observations.
        """
    if np.all(self.df_resid + self.df_model == 0.0):
        return np.full_like(self.centered_tss, np.nan)
    if self.k_constant:
        return self.centered_tss / (self.df_resid + self.df_model)
    else:
        return self.uncentered_tss / (self.df_resid + self.df_model)