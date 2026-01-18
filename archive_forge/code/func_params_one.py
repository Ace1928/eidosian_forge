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
def params_one(self):
    """Parameter estimate based on one-step approximation.

        This the one step parameter estimate computed as
        ``params`` from the full sample minus ``d_params``.
        """
    return self.results.params - self.d_params