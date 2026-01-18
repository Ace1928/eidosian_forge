import numpy as np
import shapely
def plot_line(line, ax=None, add_points=True, color=None, linewidth=2, **kwargs):
    """
    Plot a (Multi)LineString/LinearRing.

    Note: this function is experimental, and mainly targetting (interactive)
    exploration, debugging and illustration purposes.

    Parameters
    ----------
    line : shapely.LineString or shapely.LinearRing
    ax : matplotlib Axes, default None
        The axes on which to draw the plot. If not specified, will get the
        current active axes or create a new figure.
    add_points : bool, default True
        If True, also plot the coordinates (vertices) as points.
    color : matplotlib color specification
        Color for the line (edgecolor under the hood) and pointes.
    linewidth : float, default 2
        The line width for the polygon boundary.
    **kwargs
        Additional keyword arguments passed to the matplotlib Patch.

    Returns
    -------
    Matplotlib artist (PathPatch)
    """
    from matplotlib.patches import PathPatch
    from matplotlib.path import Path
    if ax is None:
        ax = _default_ax()
    if color is None:
        color = 'C0'
    if isinstance(line, shapely.MultiLineString):
        path = Path.make_compound_path(*[Path(np.asarray(mline.coords)[:, :2]) for mline in line.geoms])
    else:
        path = Path(np.asarray(line.coords)[:, :2])
    patch = PathPatch(path, facecolor='none', edgecolor=color, linewidth=linewidth, **kwargs)
    ax.add_patch(patch)
    ax.autoscale_view()
    if add_points:
        line = plot_points(line, ax=ax, color=color)
        return (patch, line)
    return patch