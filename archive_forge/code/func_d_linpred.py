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
def d_linpred(self):
    """
        Change in linear prediction

        This uses one-step approximation of the parameter change to deleting
        one observation ``d_params``.
        """
    exog = self.results.model.exog
    return (exog * self.d_params).sum(1)