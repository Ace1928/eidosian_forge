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
def cov_ratio(self):
    """covariance ratio between LOOO and original

        This uses determinant of the estimate of the parameter covariance
        from leave-one-out estimates.
        requires leave one out loop for observations
        """
    cov_ratio = self.det_cov_params_not_obsi / np.linalg.det(self.results.cov_params())
    return cov_ratio