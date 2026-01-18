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
def addtag(code, dtype, count, value, writeonce=False):
    code = int(TIFF.TAG_NAMES.get(code, code))
    try:
        tifftype = TIFF.DATA_DTYPES[dtype]
    except KeyError:
        raise ValueError('unknown dtype %s' % dtype)
    rawcount = count
    if dtype == 's':
        value = bytestr(value) + b'\x00'
        count = rawcount = len(value)
        rawcount = value.find(b'\x00\x00')
        if rawcount < 0:
            rawcount = count
        else:
            rawcount += 1
        value = (value,)
    elif isinstance(value, bytes):
        dtsize = struct.calcsize(dtype)
        if len(value) % dtsize:
            raise ValueError('invalid packed binary data')
        count = len(value) // dtsize
    if len(dtype) > 1:
        count *= int(dtype[:-1])
        dtype = dtype[-1]
    ifdentry = [pack('HH', code, tifftype), pack(offsetformat, rawcount)]
    ifdvalue = None
    if struct.calcsize(dtype) * count <= offsetsize:
        if isinstance(value, bytes):
            ifdentry.append(pack(valueformat, value))
        elif count == 1:
            if isinstance(value, (tuple, list, numpy.ndarray)):
                value = value[0]
            ifdentry.append(pack(valueformat, pack(dtype, value)))
        else:
            ifdentry.append(pack(valueformat, pack(str(count) + dtype, *value)))
    else:
        ifdentry.append(pack(offsetformat, 0))
        if isinstance(value, bytes):
            ifdvalue = value
        elif isinstance(value, numpy.ndarray):
            assert value.size == count
            assert value.dtype.char == dtype
            ifdvalue = value.tostring()
        elif isinstance(value, (tuple, list)):
            ifdvalue = pack(str(count) + dtype, *value)
        else:
            ifdvalue = pack(dtype, value)
    tags.append((code, b''.join(ifdentry), ifdvalue, writeonce))