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
def hat_matrix_diag(self):
    """
        Diagonal of the hat_matrix for GLM

        Notes
        -----
        This returns the diagonal of the hat matrix that was provided as
        argument to GLMInfluence or computes it using the results method
        `get_hat_matrix`.
        """
    if hasattr(self, '_hat_matrix_diag'):
        return self._hat_matrix_diag
    else:
        return self.results.get_hat_matrix()