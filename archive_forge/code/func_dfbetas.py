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
def dfbetas(self):
    """dfbetas

        uses results from leave-one-observation-out loop
        """
    dfbetas = self.results.params - self.params_not_obsi
    dfbetas /= np.sqrt(self.sigma2_not_obsi[:, None])
    dfbetas /= np.sqrt(np.diag(self.results.normalized_cov_params))
    return dfbetas