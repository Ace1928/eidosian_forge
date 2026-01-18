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
def reshape_nd(data_or_shape, ndim):
    """Return image array or shape with at least ndim dimensions.

    Prepend 1s to image shape as necessary.

    >>> reshape_nd(numpy.empty(0), 1).shape
    (0,)
    >>> reshape_nd(numpy.empty(1), 2).shape
    (1, 1)
    >>> reshape_nd(numpy.empty((2, 3)), 3).shape
    (1, 2, 3)
    >>> reshape_nd(numpy.empty((3, 4, 5)), 3).shape
    (3, 4, 5)
    >>> reshape_nd((2, 3), 3)
    (1, 2, 3)

    """
    is_shape = isinstance(data_or_shape, tuple)
    shape = data_or_shape if is_shape else data_or_shape.shape
    if len(shape) >= ndim:
        return data_or_shape
    shape = (1,) * (ndim - len(shape)) + shape
    return shape if is_shape else data_or_shape.reshape(shape)