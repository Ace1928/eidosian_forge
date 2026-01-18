import os
from collections.abc import Sequence
from numbers import Integral
import numpy as np
from .. import draw
from skimage import morphology
def octagon(m, n, dtype=np.uint8, *, decomposition=None):
    """Generates an octagon shaped footprint.

    For a given size of (m) horizontal and vertical sides
    and a given (n) height or width of slanted sides octagon is generated.
    The slanted sides are 45 or 135 degrees to the horizontal axis
    and hence the widths and heights are equal. The overall size of the
    footprint along a single axis will be ``m + 2 * n``.

    Parameters
    ----------
    m : int
        The size of the horizontal and vertical sides.
    n : int
        The height or width of the slanted sides.

    Other Parameters
    ----------------
    dtype : data-type, optional
        The data type of the footprint.
    decomposition : {None, 'sequence'}, optional
        If None, a single array is returned. For 'sequence', a tuple of smaller
        footprints is returned. Applying this series of smaller footprints will
        given an identical result to a single, larger footprint, but with
        better computational performance. See Notes for more details.

    Returns
    -------
    footprint : ndarray or tuple
        The footprint where elements of the neighborhood are 1 and 0 otherwise.
        When `decomposition` is None, this is just a numpy.ndarray. Otherwise,
        this will be a tuple whose length is equal to the number of unique
        structuring elements to apply (see Notes for more detail)

    Notes
    -----
    When `decomposition` is not None, each element of the `footprint`
    tuple is a 2-tuple of the form ``(ndarray, num_iter)`` that specifies a
    footprint array and the number of iterations it is to be applied.

    For either binary or grayscale morphology, using
    ``decomposition='sequence'`` was observed to have a performance benefit,
    with the magnitude of the benefit increasing with increasing footprint
    size.
    """
    if m == n == 0:
        raise ValueError('m and n cannot both be zero')
    if decomposition is None:
        from . import convex_hull_image
        footprint = np.zeros((m + 2 * n, m + 2 * n))
        footprint[0, n] = 1
        footprint[n, 0] = 1
        footprint[0, m + n - 1] = 1
        footprint[m + n - 1, 0] = 1
        footprint[-1, n] = 1
        footprint[n, -1] = 1
        footprint[-1, m + n - 1] = 1
        footprint[m + n - 1, -1] = 1
        footprint = convex_hull_image(footprint).astype(dtype)
    elif decomposition == 'sequence':
        if m <= 2 and n <= 2:
            return ((octagon(m, n, dtype=dtype, decomposition=None), 1),)
        if m == 0:
            m = 2
            n -= 1
        sequence = []
        if m > 1:
            sequence += list(square(m, dtype=dtype, decomposition='sequence'))
        if n > 0:
            sequence += [(diamond(1, dtype=dtype, decomposition=None), n)]
        footprint = tuple(sequence)
    else:
        raise ValueError(f'Unrecognized decomposition: {decomposition}')
    return footprint