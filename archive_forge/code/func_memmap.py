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
def memmap(filename, shape=None, dtype=None, page=None, series=0, mode='r+', **kwargs):
    """Return memory-mapped numpy array stored in TIFF file.

    Memory-mapping requires data stored in native byte order, without tiling,
    compression, predictors, etc.
    If 'shape' and 'dtype' are provided, existing files will be overwritten or
    appended to depending on the 'append' parameter.
    Otherwise the image data of a specified page or series in an existing
    file will be memory-mapped. By default, the image data of the first page
    series is memory-mapped.
    Call flush() to write any changes in the array to the file.
    Raise ValueError if the image data in the file is not memory-mappable.

    Parameters
    ----------
    filename : str
        Name of the TIFF file which stores the array.
    shape : tuple
        Shape of the empty array.
    dtype : numpy.dtype
        Data-type of the empty array.
    page : int
        Index of the page which image data to memory-map.
    series : int
        Index of the page series which image data to memory-map.
    mode : {'r+', 'r', 'c'}, optional
        The file open mode. Default is to open existing file for reading and
        writing ('r+').
    kwargs : dict
        Additional parameters passed to imsave() or TiffFile().

    Examples
    --------
    >>> # create an empty TIFF file and write to memory-mapped image
    >>> im = memmap('temp.tif', shape=(256, 256), dtype='float32')
    >>> im[255, 255] = 1.0
    >>> im.flush()
    >>> im.shape, im.dtype
    ((256, 256), dtype('float32'))
    >>> del im

    >>> # memory-map image data in a TIFF file
    >>> im = memmap('temp.tif', page=0)
    >>> im[255, 255]
    1.0

    """
    if shape is not None and dtype is not None:
        kwargs.update(data=None, shape=shape, dtype=dtype, returnoffset=True, align=TIFF.ALLOCATIONGRANULARITY)
        result = imsave(filename, **kwargs)
        if result is None:
            raise ValueError('image data are not memory-mappable')
        offset = result[0]
    else:
        with TiffFile(filename, **kwargs) as tif:
            if page is not None:
                page = tif.pages[page]
                if not page.is_memmappable:
                    raise ValueError('image data are not memory-mappable')
                offset, _ = page.is_contiguous
                shape = page.shape
                dtype = page.dtype
            else:
                series = tif.series[series]
                if series.offset is None:
                    raise ValueError('image data are not memory-mappable')
                shape = series.shape
                dtype = series.dtype
                offset = series.offset
            dtype = tif.byteorder + dtype.char
    return numpy.memmap(filename, dtype, mode, offset, shape, 'C')