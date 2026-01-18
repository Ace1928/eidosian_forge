from __future__ import division, print_function
import sys
import os
import io
import re
import glob
import math
import zlib
import time
import json
import enum
import struct
import pathlib
import warnings
import binascii
import tempfile
import datetime
import threading
import collections
import multiprocessing
import concurrent.futures
import numpy
def decode_floats(data):
    """Decode floating point horizontal differencing.

    The TIFF predictor type 3 reorders the bytes of the image values and
    applies horizontal byte differencing to improve compression of floating
    point images. The ordering of interleaved color channels is preserved.

    Parameters
    ----------
    data : numpy.ndarray
        The image to be decoded. The dtype must be a floating point.
        The shape must include the number of contiguous samples per pixel
        even if 1.

    """
    shape = data.shape
    dtype = data.dtype
    if len(shape) < 3:
        raise ValueError('invalid data shape')
    if dtype.char not in 'dfe':
        raise ValueError('not a floating point image')
    littleendian = data.dtype.byteorder == '<' or (sys.byteorder == 'little' and data.dtype.byteorder == '=')
    data = data.view('uint8')
    data.shape = shape[:-2] + (-1,) + shape[-1:]
    numpy.cumsum(data, axis=-2, dtype='uint8', out=data)
    if littleendian:
        data.shape = shape[:-2] + (-1,) + shape[-2:]
    data = numpy.swapaxes(data, -3, -2)
    data = numpy.swapaxes(data, -2, -1)
    data = data[..., ::-1]
    data = numpy.ascontiguousarray(data)
    data = data.view(dtype)
    data.shape = shape
    return data