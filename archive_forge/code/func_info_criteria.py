from __future__ import annotations
from statsmodels.compat.python import lrange
from collections import defaultdict
from io import StringIO
import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.base.wrapper as wrap
from statsmodels.iolib.table import SimpleTable
from statsmodels.tools.decorators import cache_readonly, deprecated_alias
from statsmodels.tools.linalg import logdet_symm
from statsmodels.tools.sm_exceptions import OutputWarning
from statsmodels.tools.validation import array_like
from statsmodels.tsa.base.tsa_model import (
import statsmodels.tsa.tsatools as tsa
from statsmodels.tsa.tsatools import duplication_matrix, unvec, vec
from statsmodels.tsa.vector_ar import output, plotting, util
from statsmodels.tsa.vector_ar.hypothesis_test_results import (
from statsmodels.tsa.vector_ar.irf import IRAnalysis
from statsmodels.tsa.vector_ar.output import VARSummary
@cache_readonly
def info_criteria(self):
    """information criteria for lagorder selection"""
    nobs = self.nobs
    neqs = self.neqs
    lag_order = self.k_ar
    free_params = lag_order * neqs ** 2 + neqs * self.k_exog
    if self.df_resid:
        ld = logdet_symm(self.sigma_u_mle)
    else:
        ld = -np.inf
    aic = ld + 2.0 / nobs * free_params
    bic = ld + np.log(nobs) / nobs * free_params
    hqic = ld + 2.0 * np.log(np.log(nobs)) / nobs * free_params
    if self.df_resid:
        fpe = ((nobs + self.df_model) / self.df_resid) ** neqs * np.exp(ld)
    else:
        fpe = np.inf
    return {'aic': aic, 'bic': bic, 'hqic': hqic, 'fpe': fpe}