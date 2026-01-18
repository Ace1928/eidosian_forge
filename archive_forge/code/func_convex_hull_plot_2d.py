import numpy as np
from scipy._lib.decorator import decorator as _decorator
@_held_figure
def convex_hull_plot_2d(hull, ax=None):
    """
    Plot the given convex hull diagram in 2-D

    Parameters
    ----------
    hull : scipy.spatial.ConvexHull instance
        Convex hull to plot
    ax : matplotlib.axes.Axes instance, optional
        Axes to plot on

    Returns
    -------
    fig : matplotlib.figure.Figure instance
        Figure for the plot

    See Also
    --------
    ConvexHull

    Notes
    -----
    Requires Matplotlib.


    Examples
    --------

    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from scipy.spatial import ConvexHull, convex_hull_plot_2d

    The convex hull of a random set of points:

    >>> rng = np.random.default_rng()
    >>> points = rng.random((30, 2))
    >>> hull = ConvexHull(points)

    Plot it:

    >>> _ = convex_hull_plot_2d(hull)
    >>> plt.show()

    """
    from matplotlib.collections import LineCollection
    if hull.points.shape[1] != 2:
        raise ValueError('Convex hull is not 2-D')
    ax.plot(hull.points[:, 0], hull.points[:, 1], 'o')
    line_segments = [hull.points[simplex] for simplex in hull.simplices]
    ax.add_collection(LineCollection(line_segments, colors='k', linestyle='solid'))
    _adjust_bounds(ax, hull.points)
    return ax.figure