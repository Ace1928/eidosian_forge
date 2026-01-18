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
def match_string(h0, n0):
    h = 0
    n = 0
    e = h
    delta_h, hc = chartocanon(h0[h:])
    h += delta_h
    delta_n, nc = chartocanon(n0[n:])
    n += delta_n
    while hc == nc:
        e = h
        if hc == ord(' '):
            while 1:
                delta_h, hc = chartocanon(h0[h:])
                h += delta_h
                if hc != ord(' '):
                    break
        else:
            delta_h, hc = chartocanon(h0[h:])
            h += delta_h
        if nc == ord(' '):
            while 1:
                delta_n, nc = chartocanon(n0[n:])
                n += delta_n
                if nc != ord(' '):
                    break
        else:
            delta_n, nc = chartocanon(n0[n:])
            n += delta_n
    return None if nc != 0 else e