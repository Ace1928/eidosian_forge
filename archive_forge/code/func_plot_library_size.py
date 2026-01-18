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
def plot_library_size(data, bins=100, log=True, cutoff=None, percentile=None, ax=None, figsize=None, xlabel='Library size', title=None, fontsize=None, filename=None, dpi=None, **kwargs):
    """Plot the library size histogram.

    Parameters
    ----------
    data : array-like, shape=[n_samples, n_features]
        Input data. Multiple datasets may be given as a list of array-likes.
    bins : int, optional (default: 100)
        Number of bins to draw in the histogram
    log : bool, or {'x', 'y'}, optional (default: True)
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
    data = utils.to_array_or_spmatrix(data)
    if not sparse.issparse(data) and (len(data.shape) > 2 or data.dtype.type is np.object_):
        libsize = [measure.library_size(d) for d in data]
    else:
        libsize = measure.library_size(data)
    return histogram(libsize, cutoff=cutoff, percentile=percentile, bins=bins, log=log, ax=ax, figsize=figsize, xlabel=xlabel, title=title, fontsize=fontsize, filename=filename, dpi=dpi, **kwargs)