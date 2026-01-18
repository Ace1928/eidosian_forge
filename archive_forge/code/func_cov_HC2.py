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
def cov_HC2(self):
    """
        Heteroscedasticity robust covariance matrix. See HC2_se.
        """
    wexog = self.model.wexog
    h = self._abat_diagonal(wexog, self.normalized_cov_params)
    self.het_scale = self.wresid ** 2 / (1 - h)
    cov_HC2 = self._HCCM(self.het_scale)
    return cov_HC2