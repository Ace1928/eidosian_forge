from functools import reduce
import numpy as np
def obliquity(affine):
    """Estimate the *obliquity* an affine's axes represent

    The term *obliquity* is defined here as the rotation of those axes with
    respect to the cardinal axes.
    This implementation is inspired by `AFNI's implementation
    <https://github.com/afni/afni/blob/b6a9f7a21c1f3231ff09efbd861f8975ad48e525/src/thd_coords.c#L660-L698>`_.
    For further details about *obliquity*, check `AFNI's documentation
    <https://sscc.nimh.nih.gov/sscc/dglen/Obliquity>`_.

    Parameters
    ----------
    affine : 2D array-like
        Affine transformation array.  Usually shape (4, 4), but can be any 2D
        array.

    Returns
    -------
    angles : 1D array-like
        The *obliquity* of each axis with respect to the cardinal axes, in radians.

    """
    vs = voxel_sizes(affine)
    best_cosines = np.abs(affine[:-1, :-1] / vs).max(axis=1)
    return np.arccos(best_cosines)