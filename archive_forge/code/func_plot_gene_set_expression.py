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
def plot_gene_set_expression(data, genes=None, starts_with=None, ends_with=None, exact_word=None, regex=None, bins=100, log=False, cutoff=None, percentile=None, library_size_normalize=False, ax=None, figsize=None, xlabel='Gene expression', title=None, fontsize=None, filename=None, dpi=None, **kwargs):
    """Plot the histogram of the expression of a gene set.

    Parameters
    ----------
    data : array-like, shape=[n_samples, n_features]
        Input data. Multiple datasets may be given as a list of array-likes.
    genes : list-like, optional (default: None)
        Integer column indices or string gene names included in gene set
    starts_with : str or None, optional (default: None)
        If not None, select genes that start with this prefix
    ends_with : str or None, optional (default: None)
        If not None, select genes that end with this suffix
    exact_word : str, list-like or None, optional (default: None)
        If not None, select genes that contain this exact word.
    regex : str or None, optional (default: None)
        If not None, select genes that match this regular expression
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
    library_size_normalize : bool, optional (default: False)
        Divide gene set expression by library size
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
    if hasattr(data, 'shape') and len(data.shape) == 2:
        expression = measure.gene_set_expression(data, genes=genes, starts_with=starts_with, ends_with=ends_with, exact_word=exact_word, regex=regex, library_size_normalize=library_size_normalize)
    else:
        data_array = utils.to_array_or_spmatrix(data)
        if len(data_array.shape) == 2 and data_array.dtype.type is not np.object_:
            expression = measure.gene_set_expression(data, genes=genes, starts_with=starts_with, ends_with=ends_with, regex=regex, library_size_normalize=library_size_normalize)
        else:
            expression = [measure.gene_set_expression(d, genes=genes, starts_with=starts_with, ends_with=ends_with, regex=regex, library_size_normalize=library_size_normalize) for d in data]
    return histogram(expression, cutoff=cutoff, percentile=percentile, bins=bins, log=log, ax=ax, figsize=figsize, xlabel=xlabel, title=title, fontsize=fontsize, filename=filename, dpi=dpi, **kwargs)