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
def read_lsm_scaninfo(fh):
    """Read LSM ScanInfo structure from file and return as dict."""
    block = {}
    blocks = [block]
    unpack = struct.unpack
    if struct.unpack('<I', fh.read(4))[0] != 268435456:
        warnings.warn('invalid LSM ScanInfo structure')
        return block
    fh.read(8)
    while True:
        entry, dtype, size = unpack('<III', fh.read(12))
        if dtype == 2:
            value = bytes2str(stripnull(fh.read(size)))
        elif dtype == 4:
            value = unpack('<i', fh.read(4))[0]
        elif dtype == 5:
            value = unpack('<d', fh.read(8))[0]
        else:
            value = 0
        if entry in TIFF.CZ_LSMINFO_SCANINFO_ARRAYS:
            blocks.append(block)
            name = TIFF.CZ_LSMINFO_SCANINFO_ARRAYS[entry]
            newobj = []
            block[name] = newobj
            block = newobj
        elif entry in TIFF.CZ_LSMINFO_SCANINFO_STRUCTS:
            blocks.append(block)
            newobj = {}
            block.append(newobj)
            block = newobj
        elif entry in TIFF.CZ_LSMINFO_SCANINFO_ATTRIBUTES:
            name = TIFF.CZ_LSMINFO_SCANINFO_ATTRIBUTES[entry]
            block[name] = value
        elif entry == 4294967295:
            block = blocks.pop()
        else:
            block['Entry0x%x' % entry] = value
        if not blocks:
            break
    return block