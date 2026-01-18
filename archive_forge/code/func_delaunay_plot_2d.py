import numpy as np
from scipy._lib.decorator import decorator as _decorator
@_held_figure
def delaunay_plot_2d(tri, ax=None):
    """
    Plot the given Delaunay triangulation in 2-D

    Parameters
    ----------
    tri : scipy.spatial.Delaunay instance
        Triangulation to plot
    ax : matplotlib.axes.Axes instance, optional
        Axes to plot on

    Returns
    -------
    fig : matplotlib.figure.Figure instance
        Figure for the plot

    See Also
    --------
    Delaunay
    matplotlib.pyplot.triplot

    Notes
    -----
    Requires Matplotlib.

    Examples
    --------

    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from scipy.spatial import Delaunay, delaunay_plot_2d

    The Delaunay triangulation of a set of random points:

    >>> rng = np.random.default_rng()
    >>> points = rng.random((30, 2))
    >>> tri = Delaunay(points)

    Plot it:

    >>> _ = delaunay_plot_2d(tri)
    >>> plt.show()

    """
    if tri.points.shape[1] != 2:
        raise ValueError('Delaunay triangulation is not 2-D')
    x, y = tri.points.T
    ax.plot(x, y, 'o')
    ax.triplot(x, y, tri.simplices.copy())
    _adjust_bounds(ax, tri.points)
    return ax.figure