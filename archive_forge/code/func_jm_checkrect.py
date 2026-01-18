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
def jm_checkrect(dev):
    """
    Check whether the last 3 path items represent a rectangle.
    Returns 1 if we have modified the path, otherwise 0.
    """
    dev.linecount = 0
    orientation = 0
    items = dev.pathdict[dictkey_items]
    len_ = len(items)
    line0 = items[len_ - 3]
    ll = JM_point_from_py(line0[1])
    lr = JM_point_from_py(line0[2])
    line2 = items[len_ - 1]
    ur = JM_point_from_py(line2[1])
    ul = JM_point_from_py(line2[2])
    if 0 or ll.y != lr.y or ll.x != ul.x or (ur.y != ul.y) or (ur.x != lr.x):
        return 0
    if ul.y < lr.y:
        r = mupdf.fz_make_rect(ul.x, ul.y, lr.x, lr.y)
        orientation = 1
    else:
        r = mupdf.fz_make_rect(ll.x, ll.y, ur.x, ur.y)
        orientation = -1
    rect = ('re', JM_py_from_rect(r), orientation)
    items[len_ - 3] = rect
    del items[len_ - 2:len_]
    return 1