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
def read_lsm_channelcolors(fh):
    """Read LSM ChannelColors structure from file and return as dict."""
    result = {'Mono': False, 'Colors': [], 'ColorNames': []}
    pos = fh.tell()
    size, ncolors, nnames, coffset, noffset, mono = struct.unpack('<IIIIII', fh.read(24))
    if ncolors != nnames:
        warnings.warn('invalid LSM ChannelColors structure')
        return result
    result['Mono'] = bool(mono)
    fh.seek(pos + coffset)
    colors = fh.read_array('uint8', count=ncolors * 4).reshape((ncolors, 4))
    result['Colors'] = colors.tolist()
    fh.seek(pos + noffset)
    buffer = fh.read(size - noffset)
    names = []
    while len(buffer) > 4:
        size = struct.unpack('<I', buffer[:4])[0]
        names.append(bytes2str(buffer[4:3 + size]))
        buffer = buffer[4 + size:]
    result['ColorNames'] = names
    return result