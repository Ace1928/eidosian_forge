import os
import sys
import ctypes
import threading
import logging
import numpy
from ..core import (
def get_image_data(self):
    dtype, shape, bpp = self._get_type_and_shape()
    array = self._wrap_bitmap_bits_in_array(shape, dtype, False)
    with self._fi as lib:
        isle = lib.FreeImage_IsLittleEndian()

    def n(arr):
        if arr.ndim == 1:
            return arr[::-1].T
        elif arr.ndim == 2:
            return arr[:, ::-1].T
        elif arr.ndim == 3:
            return arr[:, :, ::-1].T
        elif arr.ndim == 4:
            return arr[:, :, :, ::-1].T
    if len(shape) == 3 and isle and (dtype.type == numpy.uint8):
        b = n(array[0])
        g = n(array[1])
        r = n(array[2])
        if shape[0] == 3:
            return numpy.dstack((r, g, b))
        elif shape[0] == 4:
            a = n(array[3])
            return numpy.dstack((r, g, b, a))
        else:
            raise ValueError('Cannot handle images of shape %s' % shape)
    a = n(array).copy()
    return a