import numpy as np
import numpy.linalg as npl
from .deprecated import deprecate_with_version
def io_orientation(affine, tol=None):
    """Orientation of input axes in terms of output axes for `affine`

    Valid for an affine transformation from ``p`` dimensions to ``q``
    dimensions (``affine.shape == (q + 1, p + 1)``).

    The calculated orientations can be used to transform associated
    arrays to best match the output orientations. If ``p`` > ``q``, then
    some of the output axes should be considered dropped in this
    orientation.

    Parameters
    ----------
    affine : (q+1, p+1) ndarray-like
       Transformation affine from ``p`` inputs to ``q`` outputs.  Usually this
       will be a shape (4,4) matrix, transforming 3 inputs to 3 outputs, but
       the code also handles the more general case
    tol : {None, float}, optional
       threshold below which SVD values of the affine are considered zero. If
       `tol` is None, and ``S`` is an array with singular values for `affine`,
       and ``eps`` is the epsilon value for datatype of ``S``, then `tol` set
       to ``S.max() * max((q, p)) * eps``

    Returns
    -------
    orientations : (p, 2) ndarray
       one row per input axis, where the first value in each row is the closest
       corresponding output axis. The second value in each row is 1 if the
       input axis is in the same direction as the corresponding output axis and
       -1 if it is in the opposite direction.  If a row is [np.nan, np.nan],
       which can happen when p > q, then this row should be considered dropped.
    """
    affine = np.asarray(affine)
    q, p = (affine.shape[0] - 1, affine.shape[1] - 1)
    RZS = affine[:q, :p]
    zooms = np.sqrt(np.sum(RZS * RZS, axis=0))
    zooms[zooms == 0] = 1
    RS = RZS / zooms
    P, S, Qs = npl.svd(RS, full_matrices=False)
    if tol is None:
        tol = S.max() * max(RS.shape) * np.finfo(S.dtype).eps
    keep = S > tol
    R = np.dot(P[:, keep], Qs[keep])
    ornt = np.ones((p, 2), dtype=np.int8) * np.nan
    for in_ax in range(p):
        col = R[:, in_ax]
        if not np.allclose(col, 0):
            out_ax = np.argmax(np.abs(col))
            ornt[in_ax, 0] = out_ax
            assert col[out_ax] != 0
            if col[out_ax] < 0:
                ornt[in_ax, 1] = -1
            else:
                ornt[in_ax, 1] = 1
            R[out_ax, :] = 0
    return ornt