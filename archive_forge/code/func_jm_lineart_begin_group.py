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
def jm_lineart_begin_group(dev, ctx, bbox, cs, isolated, knockout, blendmode, alpha):
    if not dev.clips:
        return
    dev.pathdict = {'type': 'group', 'rect': JM_py_from_rect(bbox), 'isolated': bool(isolated), 'knockout': bool(knockout), 'blendmode': mupdf.fz_blendmode_name(blendmode), 'opacity': alpha, 'level': dev.depth, 'layer': dev.layer_name}
    jm_append_merge(dev)
    dev.depth += 1