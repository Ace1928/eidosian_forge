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
def hat_matrix_exog_diag(self):
    """Diagonal of the hat_matrix using only exog as in OLS

        """
    get_exogs = getattr(self.results.model, '_get_exogs', None)
    if get_exogs is not None:
        exog = np.column_stack(get_exogs())
    else:
        exog = self.exog
    return (exog * np.linalg.pinv(exog).T).sum(1)