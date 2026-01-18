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
def d_params(self):
    """Change in parameter estimates

        Notes
        -----
        This uses one-step approximation of the parameter change to deleting
        one observation.
        """
    beta_i = np.linalg.pinv(self.exog) * self.resid_studentized
    beta_i /= np.sqrt(1 - self.hat_matrix_diag)
    return beta_i.T