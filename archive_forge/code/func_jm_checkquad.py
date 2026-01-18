import atexit
import binascii
import collections
import glob
import inspect
import io
import math
import os
import pathlib
import re
import string
import sys
import tarfile
import typing
import warnings
import weakref
import zipfile
from . import extra
from . import _extra
from . import utils
from .table import find_tables
def jm_checkquad(dev):
    """
    Check whether the last 4 lines represent a quad.
    Because of how we count, the lines are a polyline already, i.e. last point
    of a line equals 1st point of next line.
    So we check for a polygon (last line's end point equals start point).
    If not true we return 0.
    """
    items = dev.pathdict[dictkey_items]
    len_ = len(items)
    f = [0] * 8
    for i in range(4):
        line = items[len_ - 4 + i]
        temp = JM_point_from_py(line[1])
        f[i * 2] = temp.x
        f[i * 2 + 1] = temp.y
        lp = JM_point_from_py(line[2])
    if lp.x != f[0] or lp.y != f[1]:
        return 0
    dev.linecount = 0
    q = mupdf.fz_make_quad(f[0], f[1], f[6], f[7], f[2], f[3], f[4], f[5])
    rect = ('qu', JM_py_from_quad(q))
    items[len_ - 4] = rect
    del items[len_ - 3:len_]
    return 1