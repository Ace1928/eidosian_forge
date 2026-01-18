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
def read_uic_tag(fh, tagid, planecount, offset):
    """Read a single UIC tag value from file and return tag name and value.

    UIC1Tags use an offset.

    """

    def read_int(count=1):
        value = struct.unpack('<%iI' % count, fh.read(4 * count))
        return value[0] if count == 1 else value
    try:
        name, dtype = TIFF.UIC_TAGS[tagid]
    except IndexError:
        return ('_TagId%i' % tagid, read_int())
    Fraction = TIFF.UIC_TAGS[4][1]
    if offset:
        pos = fh.tell()
        if dtype not in (int, None):
            off = read_int()
            if off < 8:
                if dtype is str:
                    return (name, '')
                warnings.warn("invalid offset for uic tag '%s': %i" % (name, off))
                return (name, off)
            fh.seek(off)
    if dtype is None:
        name = '_' + name
        value = read_int()
    elif dtype is int:
        value = read_int()
    elif dtype is Fraction:
        value = read_int(2)
        value = value[0] / value[1]
    elif dtype is julian_datetime:
        value = julian_datetime(*read_int(2))
    elif dtype is read_uic_image_property:
        value = read_uic_image_property(fh)
    elif dtype is str:
        size = read_int()
        if 0 <= size < 2 ** 10:
            value = struct.unpack('%is' % size, fh.read(size))[0][:-1]
            value = bytes2str(stripnull(value))
        elif offset:
            value = ''
            warnings.warn("corrupt string in uic tag '%s'" % name)
        else:
            raise ValueError('invalid string size: %i' % size)
    elif dtype == '%ip':
        value = []
        for _ in range(planecount):
            size = read_int()
            if 0 <= size < 2 ** 10:
                string = struct.unpack('%is' % size, fh.read(size))[0][:-1]
                string = bytes2str(stripnull(string))
                value.append(string)
            elif offset:
                warnings.warn("corrupt string in uic tag '%s'" % name)
            else:
                raise ValueError('invalid string size: %i' % size)
    else:
        dtype = '<' + dtype
        if '%i' in dtype:
            dtype = dtype % planecount
        if '(' in dtype:
            value = fh.read_array(dtype, 1)[0]
            if value.shape[-1] == 2:
                value = value[..., 0] / value[..., 1]
        else:
            value = struct.unpack(dtype, fh.read(struct.calcsize(dtype)))
            if len(value) == 1:
                value = value[0]
    if offset:
        fh.seek(pos + 4)
    return (name, value)