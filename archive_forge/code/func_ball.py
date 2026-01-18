import os
from collections.abc import Sequence
from numbers import Integral
import numpy as np
from .. import draw
from skimage import morphology
def ball(radius, dtype=np.uint8, *, strict_radius=True, decomposition=None):
    """Generates a ball-shaped footprint.

    This is the 3D equivalent of a disk.
    A pixel is within the neighborhood if the Euclidean distance between
    it and the origin is no greater than radius.

    Parameters
    ----------
    radius : int
        The radius of the ball-shaped footprint.

    Other Parameters
    ----------------
    dtype : data-type, optional
        The data type of the footprint.
    strict_radius : bool, optional
        If False, extend the radius by 0.5. This allows the circle to expand
        further within a cube that remains of size ``2 * radius + 1`` along
        each axis. This parameter is ignored if decomposition is not None.
    decomposition : {None, 'sequence'}, optional
        If None, a single array is returned. For 'sequence', a tuple of smaller
        footprints is returned. Applying this series of smaller footprints will
        given a result equivalent to a single, larger footprint, but with
        better computational performance. For ball footprints, the sequence
        decomposition is not exactly equivalent to decomposition=None.
        See Notes for more details.

    Returns
    -------
    footprint : ndarray or tuple
        The footprint where elements of the neighborhood are 1 and 0 otherwise.

    Notes
    -----
    The disk produced by the decomposition='sequence' mode is not identical
    to that with decomposition=None. Here we extend the approach taken in [1]_
    for disks to the 3D case, using 3-dimensional extensions of the "square",
    "diamond" and "t-shaped" elements from that publication. All of these
    elementary elements have size ``(3,) * ndim``. We numerically computed the
    number of repetitions of each element that gives the closest match to the
    ball computed with kwargs ``strict_radius=False, decomposition=None``.

    Empirically, the equivalent composite footprint to the sequence
    decomposition approaches a rhombicuboctahedron (26-faces [2]_).

    References
    ----------
    .. [1] Park, H and Chin R.T. Decomposition of structuring elements for
           optimal implementation of morphological operations. In Proceedings:
           1997 IEEE Workshop on Nonlinear Signal and Image Processing, London,
           UK.
           https://www.iwaenc.org/proceedings/1997/nsip97/pdf/scan/ns970226.pdf
    .. [2] https://en.wikipedia.org/wiki/Rhombicuboctahedron
    """
    if decomposition is None:
        n = 2 * radius + 1
        Z, Y, X = np.mgrid[-radius:radius:n * 1j, -radius:radius:n * 1j, -radius:radius:n * 1j]
        s = X ** 2 + Y ** 2 + Z ** 2
        if not strict_radius:
            radius += 0.5
        return np.array(s <= radius * radius, dtype=dtype)
    elif decomposition == 'sequence':
        sequence = _nsphere_series_decomposition(radius, ndim=3, dtype=dtype)
    else:
        raise ValueError(f'Unrecognized decomposition: {decomposition}')
    return sequence