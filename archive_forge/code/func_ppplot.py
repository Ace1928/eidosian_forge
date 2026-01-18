from statsmodels.compat.python import lzip
import numpy as np
from scipy import stats
from statsmodels.distributions import ECDF
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.decorators import cache_readonly
from statsmodels.tools.tools import add_constant
from . import utils
def ppplot(self, xlabel=None, ylabel=None, line=None, other=None, ax=None, **plotkwargs):
    """
        Plot of the percentiles of x versus the percentiles of a distribution.

        Parameters
        ----------
        xlabel : str or None, optional
            User-provided labels for the x-axis. If None (default),
            other values are used depending on the status of the kwarg `other`.
        ylabel : str or None, optional
            User-provided labels for the y-axis. If None (default),
            other values are used depending on the status of the kwarg `other`.
        line : {None, "45", "s", "r", q"}, optional
            Options for the reference line to which the data is compared:

            - "45": 45-degree line
            - "s": standardized line, the expected order statistics are
              scaled by the standard deviation of the given sample and have
              the mean added to them
            - "r": A regression line is fit
            - "q": A line is fit through the quartiles.
            - None: by default no reference line is added to the plot.

        other : ProbPlot, array_like, or None, optional
            If provided, ECDF(x) will be plotted against p(x) where x are
            sorted samples from `self`. ECDF is an empirical cumulative
            distribution function estimated from `other` and
            p(x) = 0.5/n, 1.5/n, ..., (n-0.5)/n where n is the number of
            samples in `self`. If an array-object is provided, it will be
            turned into a `ProbPlot` instance default parameters. If not
            provided (default), `self.dist(x)` is be plotted against p(x).

        ax : AxesSubplot, optional
            If given, this subplot is used to plot in instead of a new figure
            being created.
        **plotkwargs
            Additional arguments to be passed to the `plot` command.

        Returns
        -------
        Figure
            If `ax` is None, the created figure.  Otherwise the figure to which
            `ax` is connected.
        """
    if other is not None:
        check_other = isinstance(other, ProbPlot)
        if not check_other:
            other = ProbPlot(other)
        p_x = self.theoretical_percentiles
        ecdf_x = ECDF(other.sample_quantiles)(self.sample_quantiles)
        fig, ax = _do_plot(p_x, ecdf_x, self.dist, ax=ax, line=line, **plotkwargs)
        if xlabel is None:
            xlabel = 'Probabilities of 2nd Sample'
        if ylabel is None:
            ylabel = 'Probabilities of 1st Sample'
    else:
        fig, ax = _do_plot(self.theoretical_percentiles, self.sample_percentiles, self.dist, ax=ax, line=line, **plotkwargs)
        if xlabel is None:
            xlabel = 'Theoretical Probabilities'
        if ylabel is None:
            ylabel = 'Sample Probabilities'
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.0])
    return fig