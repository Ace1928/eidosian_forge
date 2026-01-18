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
@property
def det_cov_params_not_obsi(self):
    """determinant of cov_params of all LOOO regressions

        uses results from leave-one-observation-out loop
        """
    return np.asarray(self._res_looo['det_cov_params'])