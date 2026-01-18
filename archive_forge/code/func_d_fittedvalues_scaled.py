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
def d_fittedvalues_scaled(self):
    """
        Change in fittedvalues scaled by standard errors.

        This uses one-step approximation of the parameter change to deleting
        one observation ``d_params``, and divides by the standard errors
        for the predicted mean provided by results.get_prediction.
        """
    return self.d_fittedvalues / self._get_prediction.se