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
def resid_studentized_external(self):
    """Studentized residuals using LOOO variance

        this uses sigma from leave-one-out estimates

        requires leave one out loop for observations
        """
    sigma_looo = np.sqrt(self.sigma2_not_obsi)
    return self.get_resid_studentized_external(sigma=sigma_looo)