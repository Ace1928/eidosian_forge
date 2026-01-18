from __future__ import division
import numpy as np
from pygsp import utils
@_plt_handle_figure
def plot_filter(filters, npoints=1000, line_width=4, x_width=3, x_size=10, plot_eigenvalues=None, show_sum=None, ax=None):
    """
    Plot the spectral response of a filter bank, a set of graph filters.

    Parameters
    ----------
    filters : Filter
        Filter bank to plot.
    npoints : int
        Number of point where the filters are evaluated.
    line_width : int
        Width of the filters plots.
    x_width : int
        Width of the X marks representing the eigenvalues.
    x_size : int
        Size of the X marks representing the eigenvalues.
    plot_eigenvalues : boolean
        To plot black X marks at all eigenvalues of the graph. You need to
        compute the Fourier basis to use this option. By default the
        eigenvalues are plot if they are contained in the Graph.
    show_sum : boolean
        To plot an extra line showing the sum of the squared magnitudes
        of the filters (default True if there is multiple filters).
    plot_name : string
        name of the plot
    save_as : str
        Whether to save the plot as save_as.png and save_as.pdf. Shown in a
        window if None (default). Only available with the matplotlib backend.
    ax : matplotlib.axes
        Axes where to draw the graph. Optional, created if not passed. Only
        available with the matplotlib backend.

    Examples
    --------
    >>> from pygsp import plotting
    >>> G = graphs.Logo()
    >>> mh = filters.MexicanHat(G)
    >>> plotting.plot_filter(mh)

    """
    G = filters.G
    if plot_eigenvalues is None:
        plot_eigenvalues = hasattr(G, '_e')
    if show_sum is None:
        show_sum = filters.Nf > 1
    if plot_eigenvalues:
        for e in G.e:
            ax.axvline(x=e, color=[0.9] * 3, linewidth=1)
    x = np.linspace(0, G.lmax, npoints)
    y = filters.evaluate(x).T
    ax.plot(x, y, linewidth=line_width)
    if show_sum:
        ax.plot(x, np.sum(y ** 2, 1), 'k', linewidth=line_width)
    ax.set_xlabel("$\\lambda$: laplacian's eigenvalues / graph frequencies")
    ax.set_ylabel('$\\hat{g}(\\lambda)$: filter response')