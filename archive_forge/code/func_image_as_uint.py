from collections import OrderedDict
import numpy as np
import os
import re
import struct
import sys
import time
import logging
def image_as_uint(im, bitdepth=None):
    """Convert the given image to uint (default: uint8)

    If the dtype already matches the desired format, it is returned
    as-is. If the image is float, and all values are between 0 and 1,
    the values are multiplied by np.power(2.0, bitdepth). In all other
    situations, the values are scaled such that the minimum value
    becomes 0 and the maximum value becomes np.power(2.0, bitdepth)-1
    (255 for 8-bit and 65535 for 16-bit).
    """
    if not bitdepth:
        bitdepth = 8
    if not isinstance(im, np.ndarray):
        raise ValueError('Image must be a numpy array')
    if bitdepth == 8:
        out_type = np.uint8
    elif bitdepth == 16:
        out_type = np.uint16
    else:
        raise ValueError('Bitdepth must be either 8 or 16')
    dtype_str1 = str(im.dtype)
    dtype_str2 = out_type.__name__
    if im.dtype == np.uint8 and bitdepth == 8 or (im.dtype == np.uint16 and bitdepth == 16):
        return im
    if dtype_str1.startswith('float') and np.nanmin(im) >= 0 and (np.nanmax(im) <= 1):
        _precision_warn(dtype_str1, dtype_str2, 'Range [0, 1].')
        im = im.astype(np.float64) * (np.power(2.0, bitdepth) - 1) + 0.499999999
    elif im.dtype == np.uint16 and bitdepth == 8:
        _precision_warn(dtype_str1, dtype_str2, 'Losing 8 bits of resolution.')
        im = np.right_shift(im, 8)
    elif im.dtype == np.uint32:
        _precision_warn(dtype_str1, dtype_str2, 'Losing {} bits of resolution.'.format(32 - bitdepth))
        im = np.right_shift(im, 32 - bitdepth)
    elif im.dtype == np.uint64:
        _precision_warn(dtype_str1, dtype_str2, 'Losing {} bits of resolution.'.format(64 - bitdepth))
        im = np.right_shift(im, 64 - bitdepth)
    else:
        mi = np.nanmin(im)
        ma = np.nanmax(im)
        if not np.isfinite(mi):
            raise ValueError('Minimum image value is not finite')
        if not np.isfinite(ma):
            raise ValueError('Maximum image value is not finite')
        if ma == mi:
            return im.astype(out_type)
        _precision_warn(dtype_str1, dtype_str2, 'Range [{}, {}].'.format(mi, ma))
        im = im.astype('float64')
        im = (im - mi) / (ma - mi) * (np.power(2.0, bitdepth) - 1) + 0.499999999
    assert np.nanmin(im) >= 0
    assert np.nanmax(im) < np.power(2.0, bitdepth)
    return im.astype(out_type)