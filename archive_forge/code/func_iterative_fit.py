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
def iterative_fit(self, maxiter=3, rtol=0.0001, **kwargs):
    """
        Perform an iterative two-stage procedure to estimate a GLS model.

        The model is assumed to have AR(p) errors, AR(p) parameters and
        regression coefficients are estimated iteratively.

        Parameters
        ----------
        maxiter : int, optional
            The number of iterations.
        rtol : float, optional
            Relative tolerance between estimated coefficients to stop the
            estimation.  Stops if max(abs(last - current) / abs(last)) < rtol.
        **kwargs
            Additional keyword arguments passed to `fit`.

        Returns
        -------
        RegressionResults
            The results computed using an iterative fit.
        """
    converged = False
    i = -1
    history = {'params': [], 'rho': [self.rho]}
    for i in range(maxiter - 1):
        if hasattr(self, 'pinv_wexog'):
            del self.pinv_wexog
        self.initialize()
        results = self.fit()
        history['params'].append(results.params)
        if i == 0:
            last = results.params
        else:
            diff = np.max(np.abs(last - results.params) / np.abs(last))
            if diff < rtol:
                converged = True
                break
            last = results.params
        self.rho, _ = yule_walker(results.resid, order=self.order, df=None)
        history['rho'].append(self.rho)
    if not converged and maxiter > 0:
        if hasattr(self, 'pinv_wexog'):
            del self.pinv_wexog
        self.initialize()
    results = self.fit(history=history, **kwargs)
    results.iter = i + 1
    if not converged:
        results.history['params'].append(results.params)
        results.iter += 1
    results.converged = converged
    return results