import numpy as np
import cartopy.crs as ccrs
def mesh_projection(projection, nx, ny, x_extents=(None, None), y_extents=(None, None)):
    """
    Return sample points in the given projection which span the entire
    projection range evenly.

    The range of the x-direction and y-direction sample points will be
    within the bounds of the projection or specified extents.

    Parameters
    ----------
    projection
        A :class:`~cartopy.crs.Projection` instance.
    nx: int
        The number of sample points in the projection x-direction.
    ny: int
        The number of sample points in the projection y-direction.
    x_extents: optional
        The (lower, upper) x-direction extent of the projection.
        Defaults to the :attr:`~cartopy.crs.Projection.x_limits`.
    y_extents: optional
        The (lower, upper) y-direction extent of the projection.
        Defaults to the :attr:`~cartopy.crs.Projection.y_limits`.

    Returns
    -------
    A tuple of three items.
        The x-direction sample points
        :class:`numpy.ndarray` of shape (nx, ny), y-direction
        sample points :class:`numpy.ndarray` of shape (nx, ny),
        and the extent of the projection range as
        ``(x-lower, x-upper, y-lower, y-upper)``.

    """

    def extent(specified, default, index):
        if specified[index] is not None:
            return specified[index]
        else:
            return default[index]
    x_lower = extent(x_extents, projection.x_limits, 0)
    x_upper = extent(x_extents, projection.x_limits, 1)
    y_lower = extent(y_extents, projection.y_limits, 0)
    y_upper = extent(y_extents, projection.y_limits, 1)
    x, xstep = np.linspace(x_lower, x_upper, nx, retstep=True, endpoint=False)
    y, ystep = np.linspace(y_lower, y_upper, ny, retstep=True, endpoint=False)
    if nx == 1 and np.isnan(xstep):
        xstep = x_upper - x_lower
    if ny == 1 and np.isnan(ystep):
        ystep = y_upper - y_lower
    x += 0.5 * xstep
    y += 0.5 * ystep
    x, y = np.meshgrid(x, y)
    return (x, y, [x_lower, x_upper, y_lower, y_upper])