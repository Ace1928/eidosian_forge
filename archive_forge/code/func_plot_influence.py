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
@Appender(_plot_influence_doc.format(**{'extra_params_doc': ''}))
def plot_influence(self, external=None, alpha=0.05, criterion='cooks', size=48, plot_alpha=0.75, ax=None, **kwargs):
    if external is None:
        external = hasattr(self, '_cache') and 'res_looo' in self._cache
    from statsmodels.graphics.regressionplots import _influence_plot
    if self.hat_matrix_diag is not None:
        res = _influence_plot(self.results, self, external=external, alpha=alpha, criterion=criterion, size=size, plot_alpha=plot_alpha, ax=ax, **kwargs)
    else:
        warnings.warn('Plot uses pearson residuals and exog hat matrix.')
        res = _influence_plot(self.results, self, external=external, alpha=alpha, criterion=criterion, size=size, leverage=self.hat_matrix_exog_diag, resid=self.resid, plot_alpha=plot_alpha, ax=ax, **kwargs)
    return res