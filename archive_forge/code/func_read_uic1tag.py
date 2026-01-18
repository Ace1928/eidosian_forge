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
def read_uic1tag(fh, byteorder, dtype, count, offsetsize, planecount=None):
    """Read MetaMorph STK UIC1Tag from file and return as dict.

    Return empty dictionary if planecount is unknown.

    """
    assert dtype in ('2I', '1I') and byteorder == '<'
    result = {}
    if dtype == '2I':
        values = fh.read_array('<u4', 2 * count).reshape(count, 2)
        result = {'ZDistance': values[:, 0] / values[:, 1]}
    elif planecount:
        for _ in range(count):
            tagid = struct.unpack('<I', fh.read(4))[0]
            if tagid in (28, 29, 37, 40, 41):
                fh.read(4)
                continue
            name, value = read_uic_tag(fh, tagid, planecount, offset=True)
            result[name] = value
    return result