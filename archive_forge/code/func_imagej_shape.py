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
def imagej_shape(shape, rgb=None):
    """Return shape normalized to 6D ImageJ hyperstack TZCYXS.

    Raise ValueError if not a valid ImageJ hyperstack shape.

    >>> imagej_shape((2, 3, 4, 5, 3), False)
    (2, 3, 4, 5, 3, 1)

    """
    shape = tuple((int(i) for i in shape))
    ndim = len(shape)
    if 1 > ndim > 6:
        raise ValueError('invalid ImageJ hyperstack: not 2 to 6 dimensional')
    if rgb is None:
        rgb = shape[-1] in (3, 4) and ndim > 2
    if rgb and shape[-1] not in (3, 4):
        raise ValueError('invalid ImageJ hyperstack: not a RGB image')
    if not rgb and ndim == 6 and (shape[-1] != 1):
        raise ValueError('invalid ImageJ hyperstack: not a non-RGB image')
    if rgb or shape[-1] == 1:
        return (1,) * (6 - ndim) + shape
    return (1,) * (5 - ndim) + shape + (1,)