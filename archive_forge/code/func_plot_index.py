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
def plot_index(self, y_var='cooks', threshold=None, title=None, ax=None, idx=None, **kwds):
    """index plot for influence attributes

        Parameters
        ----------
        y_var : str
            Name of attribute or shortcut for predefined attributes that will
            be plotted on the y-axis.
        threshold : None or float
            Threshold for adding annotation with observation labels.
            Observations for which the absolute value of the y_var is larger
            than the threshold will be annotated. Set to a negative number to
            label all observations or to a large number to have no annotation.
        title : str
            If provided, the title will replace the default "Index Plot" title.
        ax : matplolib axis instance
            The plot will be added to the `ax` if provided, otherwise a new
            figure is created.
        idx : {None, int}
            Some attributes require an additional index to select the y-var.
            In dfbetas this refers to the column indes.
        kwds : optional keywords
            Keywords will be used in the call to matplotlib scatter function.
        """
    criterion = y_var
    if threshold is None:
        threshold = 'all'
    if criterion == 'dfbeta':
        y = self.dfbetas[:, idx]
        ylabel = 'DFBETA for ' + self.results.model.exog_names[idx]
    elif criterion.startswith('cook'):
        y = self.cooks_distance[0]
        ylabel = "Cook's distance"
    elif criterion.startswith('hat') or criterion.startswith('lever'):
        y = self.hat_matrix_diag
        ylabel = 'Leverage (diagonal of hat matrix)'
    elif criterion.startswith('cook'):
        y = self.cooks_distance[0]
        ylabel = "Cook's distance"
    elif criterion.startswith('resid_stu'):
        y = self.resid_studentized
        ylabel = 'Internally Studentized Residuals'
    else:
        y = getattr(self, y_var)
        if idx is not None:
            y = y[idx]
        ylabel = y_var
    fig = self._plot_index(y, ylabel, threshold=threshold, title=title, ax=ax, **kwds)
    return fig