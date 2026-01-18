import warnings
from statsmodels.compat.pandas import Appender
from statsmodels.compat.python import lzip
from collections import defaultdict
import numpy as np
from statsmodels.graphics._regressionplots_doc import _plot_influence_doc
from statsmodels.regression.linear_model import OLS
from statsmodels.stats.multitest import multipletests
from statsmodels.tools.decorators import cache_readonly
from statsmodels.tools.tools import maybe_unwrap_results
@cache_readonly
def dffits(self):
    """
        dffits measure for influence of an observation

        based on resid_studentized_external,
        uses results from leave-one-observation-out loop

        It is recommended that observations with dffits large than a
        threshold of 2 sqrt{k / n} where k is the number of parameters, should
        be investigated.

        Returns
        -------
        dffits : float
        dffits_threshold : float

        References
        ----------
        `Wikipedia <https://en.wikipedia.org/wiki/DFFITS>`_
        """
    hii = self.hat_matrix_diag
    dffits_ = self.resid_studentized_external * np.sqrt(hii / (1 - hii))
    dffits_threshold = 2 * np.sqrt(self.k_vars * 1.0 / self.nobs)
    return (dffits_, dffits_threshold)