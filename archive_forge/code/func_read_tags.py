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
def read_tags(fh, byteorder, offsetsize, tagnames, customtags=None, maxifds=None):
    """Read tags from chain of IFDs and return as list of dicts.

    The file handle position must be at a valid IFD header.

    """
    if offsetsize == 4:
        offsetformat = byteorder + 'I'
        tagnosize = 2
        tagnoformat = byteorder + 'H'
        tagsize = 12
        tagformat1 = byteorder + 'HH'
        tagformat2 = byteorder + 'I4s'
    elif offsetsize == 8:
        offsetformat = byteorder + 'Q'
        tagnosize = 8
        tagnoformat = byteorder + 'Q'
        tagsize = 20
        tagformat1 = byteorder + 'HH'
        tagformat2 = byteorder + 'Q8s'
    else:
        raise ValueError('invalid offset size')
    if customtags is None:
        customtags = {}
    if maxifds is None:
        maxifds = 2 ** 32
    result = []
    unpack = struct.unpack
    offset = fh.tell()
    while len(result) < maxifds:
        try:
            tagno = unpack(tagnoformat, fh.read(tagnosize))[0]
            if tagno > 4096:
                raise ValueError('suspicious number of tags')
        except Exception:
            warnings.warn('corrupted tag list at offset %i' % offset)
            break
        tags = {}
        data = fh.read(tagsize * tagno)
        pos = fh.tell()
        index = 0
        for _ in range(tagno):
            code, type_ = unpack(tagformat1, data[index:index + 4])
            count, value = unpack(tagformat2, data[index + 4:index + tagsize])
            index += tagsize
            name = tagnames.get(code, str(code))
            try:
                dtype = TIFF.DATA_FORMATS[type_]
            except KeyError:
                raise TiffTag.Error('unknown tag data type %i' % type_)
            fmt = '%s%i%s' % (byteorder, count * int(dtype[0]), dtype[1])
            size = struct.calcsize(fmt)
            if size > offsetsize or code in customtags:
                offset = unpack(offsetformat, value)[0]
                if offset < 8 or offset > fh.size - size:
                    raise TiffTag.Error('invalid tag value offset %i' % offset)
                fh.seek(offset)
                if code in customtags:
                    readfunc = customtags[code][1]
                    value = readfunc(fh, byteorder, dtype, count, offsetsize)
                elif type_ == 7 or (count > 1 and dtype[-1] == 'B'):
                    value = read_bytes(fh, byteorder, dtype, count, offsetsize)
                elif code in tagnames or dtype[-1] == 's':
                    value = unpack(fmt, fh.read(size))
                else:
                    value = read_numpy(fh, byteorder, dtype, count, offsetsize)
            elif dtype[-1] == 'B' or type_ == 7:
                value = value[:size]
            else:
                value = unpack(fmt, value[:size])
            if code not in customtags and code not in TIFF.TAG_TUPLE:
                if len(value) == 1:
                    value = value[0]
            if type_ != 7 and dtype[-1] == 's' and isinstance(value, bytes):
                try:
                    value = bytes2str(stripascii(value).strip())
                except UnicodeDecodeError:
                    warnings.warn('tag %i: coercing invalid ASCII to bytes' % code)
            tags[name] = value
        result.append(tags)
        fh.seek(pos)
        offset = unpack(offsetformat, fh.read(offsetsize))[0]
        if offset == 0:
            break
        if offset >= fh.size:
            warnings.warn('invalid page offset %i' % offset)
            break
        fh.seek(offset)
    if result and maxifds == 1:
        result = result[0]
    return result