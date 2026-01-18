from statsmodels.compat.pandas import deprecate_kwarg
import calendar
import numpy as np
import pandas as pd
from statsmodels.graphics import utils
from statsmodels.tools.validation import array_like
from statsmodels.tsa.stattools import acf, pacf, ccf
def plot_ccf(x, y, *, ax=None, lags=None, negative_lags=False, alpha=0.05, use_vlines=True, adjusted=False, fft=False, title='Cross-correlation', auto_ylims=False, vlines_kwargs=None, **kwargs):
    """
    Plot the cross-correlation function

    Correlations between ``x`` and the lags of ``y`` are calculated.

    The lags are shown on the horizontal axis and the correlations
    on the vertical axis.

    Parameters
    ----------
    x, y : array_like
        Arrays of time-series values.
    ax : AxesSubplot, optional
        If given, this subplot is used to plot in, otherwise a new figure with
        one subplot is created.
    lags : {int, array_like}, optional
        An int or array of lag values, used on the horizontal axis. Uses
        ``np.arange(lags)`` when lags is an int.  If not provided,
        ``lags=np.arange(len(corr))`` is used.
    negative_lags: bool, optional
        If True, negative lags are shown on the horizontal axis.
    alpha : scalar, optional
        If a number is given, the confidence intervals for the given level are
        plotted, e.g. if alpha=.05, 95 % confidence intervals are shown.
        If None, confidence intervals are not shown on the plot.
    use_vlines : bool, optional
        If True, shows vertical lines and markers for the correlation values.
        If False, only shows markers.  The default marker is 'o'; it can
        be overridden with a ``marker`` kwarg.
    adjusted : bool
        If True, then denominators for cross-correlations are n-k, otherwise n.
    fft : bool, optional
        If True, computes the CCF via FFT.
    title : str, optional
        Title to place on plot. Default is 'Cross-correlation'.
    auto_ylims : bool, optional
        If True, adjusts automatically the vertical axis limits to CCF values.
    vlines_kwargs : dict, optional
        Optional dictionary of keyword arguments that are passed to vlines.
    **kwargs : kwargs, optional
        Optional keyword arguments that are directly passed on to the
        Matplotlib ``plot`` and ``axhline`` functions.

    Returns
    -------
    Figure
        The figure where the plot is drawn. This is either an existing figure
        if the `ax` argument is provided, or a newly created figure
        if `ax` is None.

    See Also
    --------
    See notes and references for statsmodels.graphics.tsaplots.plot_acf

    Examples
    --------
    >>> import pandas as pd
    >>> import matplotlib.pyplot as plt
    >>> import statsmodels.api as sm

    >>> dta = sm.datasets.macrodata.load_pandas().data
    >>> diffed = dta.diff().dropna()
    >>> sm.graphics.tsa.plot_ccf(diffed["unemp"], diffed["infl"])
    >>> plt.show()
    """
    fig, ax = utils.create_mpl_ax(ax)
    lags, nlags, irregular = _prepare_data_corr_plot(x, lags, True)
    vlines_kwargs = {} if vlines_kwargs is None else vlines_kwargs
    if negative_lags:
        lags = -lags
    ccf_res = ccf(x, y, adjusted=adjusted, fft=fft, alpha=alpha, nlags=nlags + 1)
    if alpha is not None:
        ccf_xy, confint = ccf_res
    else:
        ccf_xy = ccf_res
        confint = None
    _plot_corr(ax, title, ccf_xy, confint, lags, irregular, use_vlines, vlines_kwargs, auto_ylims=auto_ylims, skip_lag0_confint=False, **kwargs)
    return fig