import numpy as np
from nibabel.imageclasses import spatial_axes_first
def count_nonzero_voxels(img):
    """
    Count number of non-zero voxels

    Parameters
    ----------
    img : ``SpatialImage``
        All voxels of the mask should be of value 1, background should have value 0.

    Returns
    -------
    count : int
        Number of non-zero voxels

    """
    return np.count_nonzero(img.dataobj)