import numpy as np
import pandas as pd
from statsmodels.compat.pandas import Appender
from statsmodels.tools.data import _is_using_pandas
from statsmodels.tsa.statespace.mlemodel import (
from statsmodels.tsa.statespace.tools import concat
from statsmodels.tools.tools import Bunch
from statsmodels.tools.decorators import cache_readonly
import statsmodels.base.wrapper as wrap
def plot_cusum(self, alpha=0.05, legend_loc='upper left', fig=None, figsize=None):
    """
        Plot the CUSUM statistic and significance bounds.

        Parameters
        ----------
        alpha : float, optional
            The plotted significance bounds are alpha %.
        legend_loc : str, optional
            The location of the legend in the plot. Default is upper left.
        fig : Figure, optional
            If given, subplots are created in this figure instead of in a new
            figure. Note that the grid will be created in the provided
            figure using `fig.add_subplot()`.
        figsize : tuple, optional
            If a figure is created, this argument allows specifying a size.
            The tuple is (width, height).

        Notes
        -----
        Evidence of parameter instability may be found if the CUSUM statistic
        moves out of the significance bounds.

        References
        ----------
        .. [*] Brown, R. L., J. Durbin, and J. M. Evans. 1975.
           "Techniques for Testing the Constancy of
           Regression Relationships over Time."
           Journal of the Royal Statistical Society.
           Series B (Methodological) 37 (2): 149-92.
        """
    from statsmodels.graphics.utils import _import_mpl, create_mpl_fig
    _import_mpl()
    fig = create_mpl_fig(fig, figsize)
    ax = fig.add_subplot(1, 1, 1)
    if hasattr(self.data, 'dates') and self.data.dates is not None:
        dates = self.data.dates._mpl_repr()
    else:
        dates = np.arange(self.nobs)
    d = max(self.nobs_diffuse, self.loglikelihood_burn)
    ax.plot(dates[d:], self.cusum, label='CUSUM')
    ax.hlines(0, dates[d], dates[-1], color='k', alpha=0.3)
    lower_line, upper_line = self._cusum_significance_bounds(alpha)
    ax.plot([dates[d], dates[-1]], upper_line, 'k--', label='%d%% significance' % (alpha * 100))
    ax.plot([dates[d], dates[-1]], lower_line, 'k--')
    ax.legend(loc=legend_loc)
    return fig