import sys
import os
import random
import logging
import json
import warnings
from numbers import Number
import numpy as np
from .. import numpy as _mx_np  # pylint: disable=reimported
from ..base import numeric_types
from .. import ndarray as nd
from ..ndarray import _internal
from .. import io
from .. import recordio
from .. util import is_np_array
from ..ndarray.numpy import _internal as _npi
def copyMakeBorder(src, top, bot, left, right, *args, **kwargs):
    """Pad image border with OpenCV.

    Parameters
    ----------
    src : NDArray
        source image
    top : int, required
        Top margin.
    bot : int, required
        Bottom margin.
    left : int, required
        Left margin.
    right : int, required
        Right margin.
    type : int, optional, default='0'
        Filling type (default=cv2.BORDER_CONSTANT).
        0 - cv2.BORDER_CONSTANT - Adds a constant colored border.
        1 - cv2.BORDER_REFLECT - Border will be mirror reflection of the
        border elements, like this : fedcba|abcdefgh|hgfedcb
        2 - cv2.BORDER_REFLECT_101 or cv.BORDER_DEFAULT - Same as above,
        but with a slight change, like this : gfedcb|abcdefgh|gfedcba
        3 - cv2.BORDER_REPLICATE - Last element is replicated throughout,
        like this: aaaaaa|abcdefgh|hhhhhhh
        4 - cv2.BORDER_WRAP - it will look like this : cdefgh|abcdefgh|abcdefg
    value : double, optional, default=0
        (Deprecated! Use ``values`` instead.) Fill with single value.
    values : tuple of <double>, optional, default=[]
        Fill with value(RGB[A] or gray), up to 4 channels.

    out : NDArray, optional
        The output NDArray to hold the result.

    Returns
    -------
    out : NDArray or list of NDArrays
        The output of this function.

    Example
    --------
    >>> with open("flower.jpeg", 'rb') as fp:
    ...     str_image = fp.read()
    ...
    >>> image = mx.img.imdecode(str_image)
    >>> image
    <NDArray 2321x3482x3 @cpu(0)>
    >>> new_image = mx_border = mx.image.copyMakeBorder(mx_img, 1, 2, 3, 4, type=0)
    >>> new_image
    <NDArray 2324x3489x3 @cpu(0)>
    """
    return _internal._cvcopyMakeBorder(src, top, bot, left, right, *args, **kwargs)