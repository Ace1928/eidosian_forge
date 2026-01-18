from statsmodels.compat.python import lzip
import numpy as np
from scipy import stats
from statsmodels.distributions import ECDF
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.decorators import cache_readonly
from statsmodels.tools.tools import add_constant
from . import utils
def qqline(ax, line, x=None, y=None, dist=None, fmt='r-', **lineoptions):
    """
    Plot a reference line for a qqplot.

    Parameters
    ----------
    ax : matplotlib axes instance
        The axes on which to plot the line
    line : str {"45","r","s","q"}
        Options for the reference line to which the data is compared.:

        - "45" - 45-degree line
        - "s"  - standardized line, the expected order statistics are scaled by
                 the standard deviation of the given sample and have the mean
                 added to them
        - "r"  - A regression line is fit
        - "q"  - A line is fit through the quartiles.
        - None - By default no reference line is added to the plot.

    x : ndarray
        X data for plot. Not needed if line is "45".
    y : ndarray
        Y data for plot. Not needed if line is "45".
    dist : scipy.stats.distribution
        A scipy.stats distribution, needed if line is "q".
    fmt : str, optional
        Line format string passed to `plot`.
    **lineoptions
        Additional arguments to be passed to the `plot` command.

    Notes
    -----
    There is no return value. The line is plotted on the given `ax`.

    Examples
    --------
    Import the food expenditure dataset.  Plot annual food expenditure on x-axis
    and household income on y-axis.  Use qqline to add regression line into the
    plot.

    >>> import statsmodels.api as sm
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from statsmodels.graphics.gofplots import qqline

    >>> foodexp = sm.datasets.engel.load()
    >>> x = foodexp.exog
    >>> y = foodexp.endog
    >>> ax = plt.subplot(111)
    >>> plt.scatter(x, y)
    >>> ax.set_xlabel(foodexp.exog_name[0])
    >>> ax.set_ylabel(foodexp.endog_name)
    >>> qqline(ax, "r", x, y)
    >>> plt.show()

    .. plot:: plots/graphics_gofplots_qqplot_qqline.py
    """
    lineoptions = lineoptions.copy()
    for ls in ('-', '--', '-.', ':'):
        if ls in fmt:
            lineoptions.setdefault('linestyle', ls)
            fmt = fmt.replace(ls, '')
            break
    for marker in ('.', ',', 'o', 'v', '^', '<', '>', '1', '2', '3', '4', '8', 's', 'p', 'P', '*', 'h', 'H', '+', 'x', 'X', 'D', 'd', '|', '_'):
        if marker in fmt:
            lineoptions.setdefault('marker', marker)
            fmt = fmt.replace(marker, '')
            break
    if fmt:
        lineoptions.setdefault('color', fmt)
    if line == '45':
        end_pts = lzip(ax.get_xlim(), ax.get_ylim())
        end_pts[0] = min(end_pts[0])
        end_pts[1] = max(end_pts[1])
        ax.plot(end_pts, end_pts, **lineoptions)
        ax.set_xlim(end_pts)
        ax.set_ylim(end_pts)
        return
    if x is None or y is None:
        raise ValueError('If line is not 45, x and y cannot be None.')
    x = np.array(x)
    y = np.array(y)
    if line == 'r':
        y = OLS(y, add_constant(x)).fit().fittedvalues
        ax.plot(x, y, **lineoptions)
    elif line == 's':
        m, b = (np.std(y), np.mean(y))
        ref_line = x * m + b
        ax.plot(x, ref_line, **lineoptions)
    elif line == 'q':
        _check_for(dist, 'ppf')
        q25 = stats.scoreatpercentile(y, 25)
        q75 = stats.scoreatpercentile(y, 75)
        theoretical_quartiles = dist.ppf([0.25, 0.75])
        m = (q75 - q25) / np.diff(theoretical_quartiles)
        b = q25 - m * theoretical_quartiles[0]
        ax.plot(x, m * x + b, **lineoptions)