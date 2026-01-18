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
def jm_lineart_stroke_path(dev, ctx, path, stroke, ctm, colorspace, color, alpha, color_params):
    try:
        assert isinstance(ctm, mupdf.fz_matrix)
        dev.pathfactor = 1
        if abs(ctm.a) == abs(ctm.d):
            dev.pathfactor = abs(ctm.a)
        dev.ctm = mupdf.FzMatrix(ctm)
        dev.path_type = trace_device_STROKE_PATH
        jm_lineart_path(dev, ctx, path)
        if dev.pathdict is None:
            return
        dev.pathdict[dictkey_type] = 's'
        dev.pathdict['stroke_opacity'] = alpha
        dev.pathdict['color'] = jm_lineart_color(colorspace, color)
        dev.pathdict[dictkey_width] = dev.pathfactor * stroke.linewidth
        dev.pathdict['lineCap'] = (stroke.start_cap, stroke.dash_cap, stroke.end_cap)
        dev.pathdict['lineJoin'] = dev.pathfactor * stroke.linejoin
        if 'closePath' not in dev.pathdict:
            dev.pathdict['closePath'] = False
        if stroke.dash_len:
            buff = mupdf.fz_new_buffer(256)
            mupdf.fz_append_string(buff, '[ ')
            for i in range(stroke.dash_len):
                value = mupdf.floats_getitem(stroke.dash_list, i)
                mupdf.fz_append_string(buff, f'{dev.pathfactor * value:g} ')
            mupdf.fz_append_string(buff, f'] {dev.pathfactor * stroke.dash_phase:g}')
            dev.pathdict['dashes'] = buff
        else:
            dev.pathdict['dashes'] = '[] 0'
        dev.pathdict[dictkey_rect] = JM_py_from_rect(dev.pathrect)
        dev.pathdict['layer'] = dev.layer_name
        dev.pathdict['seqno'] = dev.seqno
        if dev.clips:
            dev.pathdict['level'] = dev.depth
        jm_append_merge(dev)
        dev.seqno += 1
    except Exception:
        if g_exceptions_verbose:
            exception_info()
        raise