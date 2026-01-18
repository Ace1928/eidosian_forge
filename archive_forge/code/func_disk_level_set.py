from itertools import cycle
import numpy as np
from scipy import ndimage as ndi
from .._shared.utils import check_nD
def disk_level_set(image_shape, *, center=None, radius=None):
    """Create a disk level set with binary values.

    Parameters
    ----------
    image_shape : tuple of positive integers
        Shape of the image
    center : tuple of positive integers, optional
        Coordinates of the center of the disk given in (row, column). If not
        given, it defaults to the center of the image.
    radius : float, optional
        Radius of the disk. If not given, it is set to the 75% of the
        smallest image dimension.

    Returns
    -------
    out : array with shape `image_shape`
        Binary level set of the disk with the given `radius` and `center`.

    See Also
    --------
    checkerboard_level_set
    """
    if center is None:
        center = tuple((i // 2 for i in image_shape))
    if radius is None:
        radius = min(image_shape) * 3.0 / 8.0
    grid = np.mgrid[[slice(i) for i in image_shape]]
    grid = (grid.T - center).T
    phi = radius - np.sqrt(np.sum(grid ** 2, 0))
    res = np.int8(phi > 0)
    return res