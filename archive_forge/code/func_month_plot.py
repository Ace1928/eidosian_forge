from statsmodels.compat.pandas import deprecate_kwarg
import calendar
import numpy as np
import pandas as pd
from statsmodels.graphics import utils
from statsmodels.tools.validation import array_like
from statsmodels.tsa.stattools import acf, pacf, ccf
def month_plot(x, dates=None, ylabel=None, ax=None):
    """
    Seasonal plot of monthly data.

    Parameters
    ----------
    x : array_like
        Seasonal data to plot. If dates is None, x must be a pandas object
        with a PeriodIndex or DatetimeIndex with a monthly frequency.
    dates : array_like, optional
        If `x` is not a pandas object, then dates must be supplied.
    ylabel : str, optional
        The label for the y-axis. Will attempt to use the `name` attribute
        of the Series.
    ax : Axes, optional
        Existing axes instance.

    Returns
    -------
    Figure
       If `ax` is provided, the Figure instance attached to `ax`. Otherwise
       a new Figure instance.

    Examples
    --------
    >>> import statsmodels.api as sm
    >>> import pandas as pd

    >>> dta = sm.datasets.elnino.load_pandas().data
    >>> dta['YEAR'] = dta.YEAR.astype(int).astype(str)
    >>> dta = dta.set_index('YEAR').T.unstack()
    >>> dates = pd.to_datetime(list(map(lambda x: '-'.join(x) + '-1',
    ...                                 dta.index.values)))
    >>> dta.index = pd.DatetimeIndex(dates, freq='MS')
    >>> fig = sm.graphics.tsa.month_plot(dta)

    .. plot:: plots/graphics_tsa_month_plot.py
    """
    if dates is None:
        from statsmodels.tools.data import _check_period_index
        _check_period_index(x, freq='M')
    else:
        x = pd.Series(x, index=pd.PeriodIndex(dates, freq='M'))
    xticklabels = list(calendar.month_abbr)[1:]
    return seasonal_plot(x.groupby(lambda y: y.month), xticklabels, ylabel=ylabel, ax=ax)