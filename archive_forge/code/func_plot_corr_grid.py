import numpy as np
from . import utils
def plot_corr_grid(dcorrs, titles=None, ncols=None, normcolor=False, xnames=None, ynames=None, fig=None, cmap='RdYlBu_r'):
    """
    Create a grid of correlation plots.

    The individual correlation plots are assumed to all have the same
    variables, axis labels can be specified only once.

    Parameters
    ----------
    dcorrs : list or iterable of ndarrays
        List of correlation matrices.
    titles : list[str], optional
        List of titles for the subplots.  By default no title are shown.
    ncols : int, optional
        Number of columns in the subplot grid.  If not given, the number of
        columns is determined automatically.
    normcolor : bool or tuple, optional
        If False (default), then the color coding range corresponds to the
        range of `dcorr`.  If True, then the color range is normalized to
        (-1, 1).  If this is a tuple of two numbers, then they define the range
        for the color bar.
    xnames : list[str], optional
        Labels for the horizontal axis.  If not given (None), then the
        matplotlib defaults (integers) are used.  If it is an empty list, [],
        then no ticks and labels are added.
    ynames : list[str], optional
        Labels for the vertical axis.  Works the same way as `xnames`.
        If not given, the same names as for `xnames` are re-used.
    fig : Figure, optional
        If given, this figure is simply returned.  Otherwise a new figure is
        created.
    cmap : str or Matplotlib Colormap instance, optional
        The colormap for the plot.  Can be any valid Matplotlib Colormap
        instance or name.

    Returns
    -------
    Figure
        If `ax` is None, the created figure.  Otherwise the figure to which
        `ax` is connected.

    Examples
    --------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> import statsmodels.api as sm

    In this example we just reuse the same correlation matrix several times.
    Of course in reality one would show a different correlation (measuring a
    another type of correlation, for example Pearson (linear) and Spearman,
    Kendall (nonlinear) correlations) for the same variables.

    >>> hie_data = sm.datasets.randhie.load_pandas()
    >>> corr_matrix = np.corrcoef(hie_data.data.T)
    >>> sm.graphics.plot_corr_grid([corr_matrix] * 8, xnames=hie_data.names)
    >>> plt.show()

    .. plot:: plots/graphics_correlation_plot_corr_grid.py
    """
    if ynames is None:
        ynames = xnames
    if not titles:
        titles = [''] * len(dcorrs)
    n_plots = len(dcorrs)
    if ncols is not None:
        nrows = int(np.ceil(n_plots / float(ncols)))
    elif n_plots < 4:
        nrows, ncols = (1, n_plots)
    else:
        nrows = int(np.sqrt(n_plots))
        ncols = int(np.ceil(n_plots / float(nrows)))
    aspect = min(ncols / float(nrows), 1.8)
    vsize = np.sqrt(nrows) * 5
    fig = utils.create_mpl_fig(fig, figsize=(vsize * aspect + 1, vsize))
    for i, c in enumerate(dcorrs):
        ax = fig.add_subplot(nrows, ncols, i + 1)
        _xnames = xnames if nrows * ncols - (i + 1) < ncols else []
        _ynames = ynames if (i + 1) % ncols == 1 else []
        plot_corr(c, xnames=_xnames, ynames=_ynames, title=titles[i], normcolor=normcolor, ax=ax, cmap=cmap)
    fig.subplots_adjust(bottom=0.1, left=0.09, right=0.9, top=0.9)
    cax = fig.add_axes([0.92, 0.1, 0.025, 0.8])
    fig.colorbar(fig.axes[0].images[0], cax=cax)
    return fig