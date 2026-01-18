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
def snipstr(string, width=79, snipat=0.5, ellipsis='...'):
    """Return string cut to specified length.

    >>> snipstr('abcdefghijklmnop', 8)
    'abc...op'

    """
    if ellipsis is None:
        if isinstance(string, bytes):
            ellipsis = b'...'
        else:
            ellipsis = '…'
    esize = len(ellipsis)
    splitlines = string.splitlines()
    result = []
    for line in splitlines:
        if line is None:
            result.append(ellipsis)
            continue
        linelen = len(line)
        if linelen <= width:
            result.append(string)
            continue
        split = snipat
        if split is None or split == 1:
            split = linelen
        elif 0 < abs(split) < 1:
            split = int(math.floor(linelen * split))
        if split < 0:
            split += linelen
            if split < 0:
                split = 0
        if esize == 0 or width < esize + 1:
            if split <= 0:
                result.append(string[-width:])
            else:
                result.append(string[:width])
        elif split <= 0:
            result.append(ellipsis + string[esize - width:])
        elif split >= linelen or width < esize + 4:
            result.append(string[:width - esize] + ellipsis)
        else:
            splitlen = linelen - width + esize
            end1 = split - splitlen // 2
            end2 = end1 + splitlen
            result.append(string[:end1] + ellipsis + string[end2:])
    if isinstance(string, bytes):
        return b'\n'.join(result)
    else:
        return '\n'.join(result)