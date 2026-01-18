from __future__ import annotations
import atexit
import builtins
import io
import logging
import math
import os
import re
import struct
import sys
import tempfile
import warnings
from collections.abc import Callable, MutableMapping
from enum import IntEnum
from pathlib import Path
from . import (
from ._binary import i32le, o32be, o32le
from ._util import DeferredError, is_path
def fromarray(obj, mode=None):
    """
    Creates an image memory from an object exporting the array interface
    (using the buffer protocol)::

      from PIL import Image
      import numpy as np
      a = np.zeros((5, 5))
      im = Image.fromarray(a)

    If ``obj`` is not contiguous, then the ``tobytes`` method is called
    and :py:func:`~PIL.Image.frombuffer` is used.

    In the case of NumPy, be aware that Pillow modes do not always correspond
    to NumPy dtypes. Pillow modes only offer 1-bit pixels, 8-bit pixels,
    32-bit signed integer pixels, and 32-bit floating point pixels.

    Pillow images can also be converted to arrays::

      from PIL import Image
      import numpy as np
      im = Image.open("hopper.jpg")
      a = np.asarray(im)

    When converting Pillow images to arrays however, only pixel values are
    transferred. This means that P and PA mode images will lose their palette.

    :param obj: Object with array interface
    :param mode: Optional mode to use when reading ``obj``. Will be determined from
      type if ``None``.

      This will not be used to convert the data after reading, but will be used to
      change how the data is read::

        from PIL import Image
        import numpy as np
        a = np.full((1, 1), 300)
        im = Image.fromarray(a, mode="L")
        im.getpixel((0, 0))  # 44
        im = Image.fromarray(a, mode="RGB")
        im.getpixel((0, 0))  # (44, 1, 0)

      See: :ref:`concept-modes` for general information about modes.
    :returns: An image object.

    .. versionadded:: 1.1.6
    """
    arr = obj.__array_interface__
    shape = arr['shape']
    ndim = len(shape)
    strides = arr.get('strides', None)
    if mode is None:
        try:
            typekey = ((1, 1) + shape[2:], arr['typestr'])
        except KeyError as e:
            msg = 'Cannot handle this data type'
            raise TypeError(msg) from e
        try:
            mode, rawmode = _fromarray_typemap[typekey]
        except KeyError as e:
            typekey_shape, typestr = typekey
            msg = f'Cannot handle this data type: {typekey_shape}, {typestr}'
            raise TypeError(msg) from e
    else:
        rawmode = mode
    if mode in ['1', 'L', 'I', 'P', 'F']:
        ndmax = 2
    elif mode == 'RGB':
        ndmax = 3
    else:
        ndmax = 4
    if ndim > ndmax:
        msg = f'Too many dimensions: {ndim} > {ndmax}.'
        raise ValueError(msg)
    size = (1 if ndim == 1 else shape[1], shape[0])
    if strides is not None:
        if hasattr(obj, 'tobytes'):
            obj = obj.tobytes()
        else:
            obj = obj.tostring()
    return frombuffer(mode, size, obj, 'raw', rawmode, 0, 1)