import math
import textwrap
from abc import ABC, abstractmethod
import numpy as np
from scipy import spatial
from .._shared.utils import safe_as_int
from .._shared.compat import NP_COPY_IF_NEEDED
def estimate_transform(ttype, src, dst, *args, **kwargs):
    """Estimate 2D geometric transformation parameters.

    You can determine the over-, well- and under-determined parameters
    with the total least-squares method.

    Number of source and destination coordinates must match.

    Parameters
    ----------
    ttype : {'euclidean', similarity', 'affine', 'piecewise-affine',              'projective', 'polynomial'}
        Type of transform.
    kwargs : array_like or int
        Function parameters (src, dst, n, angle)::

            NAME / TTYPE        FUNCTION PARAMETERS
            'euclidean'         `src, `dst`
            'similarity'        `src, `dst`
            'affine'            `src, `dst`
            'piecewise-affine'  `src, `dst`
            'projective'        `src, `dst`
            'polynomial'        `src, `dst`, `order` (polynomial order,
                                                      default order is 2)

        Also see examples below.

    Returns
    -------
    tform : :class:`_GeometricTransform`
        Transform object containing the transformation parameters and providing
        access to forward and inverse transformation functions.

    Examples
    --------
    >>> import numpy as np
    >>> import skimage as ski

    >>> # estimate transformation parameters
    >>> src = np.array([0, 0, 10, 10]).reshape((2, 2))
    >>> dst = np.array([12, 14, 1, -20]).reshape((2, 2))

    >>> tform = ski.transform.estimate_transform('similarity', src, dst)

    >>> np.allclose(tform.inverse(tform(src)), src)
    True

    >>> # warp image using the estimated transformation
    >>> image = ski.data.camera()

    >>> ski.transform.warp(image, inverse_map=tform.inverse) # doctest: +SKIP

    >>> # create transformation with explicit parameters
    >>> tform2 = ski.transform.SimilarityTransform(scale=1.1, rotation=1,
    ...     translation=(10, 20))

    >>> # unite transformations, applied in order from left to right
    >>> tform3 = tform + tform2
    >>> np.allclose(tform3(src), tform2(tform(src)))
    True

    """
    ttype = ttype.lower()
    if ttype not in TRANSFORMS:
        raise ValueError(f"the transformation type '{ttype}' is not implemented")
    tform = TRANSFORMS[ttype](dimensionality=src.shape[1])
    tform.estimate(src, dst, *args, **kwargs)
    return tform