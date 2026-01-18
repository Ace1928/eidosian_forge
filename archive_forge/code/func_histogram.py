from .. import measure
from .. import utils
from .tools import label_axis
from .utils import _get_figure
from .utils import parse_fontsize
from .utils import show
from .utils import temp_fontsize
from scipy import sparse
import numbers
import numpy as np
@utils._with_pkg(pkg='matplotlib', min_version=3)
def histogram(data, bins=100, log=False, cutoff=None, percentile=None, ax=None, figsize=None, xlabel=None, ylabel='Number of cells', title=None, fontsize=None, histtype='stepfilled', label=None, legend=True, alpha=None, filename=None, dpi=None, **kwargs):
    """Plot a histogram.

    Parameters
    ----------
    data : array-like, shape=[n_samples]
        Input data. Multiple datasets may be given as a list of array-likes.
    bins : int, optional (default: 100)
        Number of bins to draw in the histogram
    log : bool, or {'x', 'y'}, optional (default: False)
        If True, plot both axes on a log scale. If 'x' or 'y',
        only plot the given axis on a log scale. If False,
        plot both axes on a linear scale.
    cutoff : float or `None`, optional (default: `None`)
        Absolute cutoff at which to draw a vertical line.
        Only one of `cutoff` and `percentile` may be given.
    percentile : float or `None`, optional (default: `None`)
        Percentile between 0 and 100 at which to draw a vertical line.
        Only one of `cutoff` and `percentile` may be given.
    ax : `matplotlib.Axes` or None, optional (default: None)
        Axis to plot on. If None, a new axis will be created.
    figsize : tuple or None, optional (default: None)
        If not None, sets the figure size (width, height)
    [x,y]label : str, optional
        Labels to display on the x and y axis.
    title : str or None, optional (default: None)
        Axis title.
    fontsize : float or None (default: None)
        Base font size.
    histtype : {'bar', 'barstacked', 'step', 'stepfilled'}, optional
        (default: 'stepfilled')
        The type of histogram to draw.
        'bar' is a traditional bar-type histogram. If multiple data are given the bars
        are arranged side by side.
        'barstacked' is a bar-type histogram where multiple data are stacked on top of
        each other.
        'step' generates a lineplot that is by default unfilled.
        'stepfilled' generates a lineplot that is by default filled.
    label : str or None, optional (default: None)
        String, or sequence of strings to match multiple datasets.
    legend : bool, optional (default: True)
        Show the legend if ``label`` is given.
    alpha : float, optional (default: 1 for a single dataset, 0.5 for multiple)
        Histogram transparency
    filename : str or None (default: None)
        file to which the output is saved
    dpi : int or None, optional (default: None)
        The resolution in dots per inch. If None it will default to the value
        savefig.dpi in the matplotlibrc file. If 'figure' it will set the dpi
        to be the value of the figure. Only used if filename is not None.
    **kwargs : additional arguments for `matplotlib.pyplot.hist`

    Returns
    -------
    ax : `matplotlib.Axes`
        axis on which plot was drawn
    """
    with temp_fontsize(fontsize):
        fig, ax, show_fig = _get_figure(ax, figsize)
        data = utils.toarray(data).squeeze()
        if len(data.shape) > 1 or data.dtype.type is np.object_:
            data = [d for d in data]
            xmin = np.min([np.min(d) for d in data])
            xmax = np.max([np.max(d) for d in data])
            if alpha is None:
                alpha = 0.5
        else:
            xmin = np.min(data)
            xmax = np.max(data)
            if alpha is None:
                alpha = 1
        if log == 'x' or log is True:
            d_flat = np.concatenate(data) if isinstance(data, list) else data
            abs_min = np.min(np.where(d_flat != 0, np.abs(d_flat), np.max(np.abs(d_flat))))
            if abs_min == 0:
                abs_min = 0.1
            bins = _symlog_bins(xmin, xmax, abs_min, bins=bins)
        ax.hist(data, bins=bins, histtype=histtype, alpha=alpha, label=label, **kwargs)
        if log == 'x' or log is True:
            ax.set_xscale('symlog', linthresh=abs_min)
        if log == 'y' or log is True:
            ax.set_yscale('log')
        label_axis(ax.xaxis, label=xlabel)
        label_axis(ax.yaxis, label=ylabel)
        if title is not None:
            ax.set_title(title, fontsize=parse_fontsize(None, 'xx-large'))
        cutoff = utils._get_percentile_cutoff(data, cutoff, percentile, required=False)
        if cutoff is not None:
            if isinstance(cutoff, numbers.Number):
                ax.axvline(cutoff, color='red')
            else:
                for c in cutoff:
                    ax.axvline(c, color='red')
        if label is not None and legend:
            ax.legend()
        if show_fig:
            show(fig)
        if filename is not None:
            fig.savefig(filename, dpi=dpi)
    return ax