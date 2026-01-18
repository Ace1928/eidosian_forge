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
def jm_lineart_clip_stroke_path(dev, ctx, path, stroke, ctm, scissor):
    if not dev.clips:
        return
    dev.ctm = mupdf.FzMatrix(ctm)
    dev.path_type = trace_device_CLIP_STROKE_PATH
    jm_lineart_path(dev, ctx, path)
    if dev.pathdict is None:
        return
    dev.pathdict['dictkey_type'] = 'clip'
    dev.pathdict['even_odd'] = None
    if 'closePath' not in dev.pathdict:
        dev.pathdict['closePath'] = False
    dev.pathdict['scissor'] = JM_py_from_rect(compute_scissor(dev))
    dev.pathdict['level'] = dev.depth
    dev.pathdict['layer'] = dev.layer_name
    jm_append_merge(dev)
    dev.depth += 1