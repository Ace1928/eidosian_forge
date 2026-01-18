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
def on_highlight_char(hits, line, ch):
    assert hits
    assert isinstance(line, mupdf.FzStextLine)
    assert isinstance(ch, mupdf.FzStextChar)
    vfuzz = ch.m_internal.size * hits.vfuzz
    hfuzz = ch.m_internal.size * hits.hfuzz
    ch_quad = JM_char_quad(line, ch)
    if hits.len > 0:
        quad = hits.quads[hits.len - 1]
        end = JM_quad_from_py(quad)
        if 1 and hdist(line.m_internal.dir, end.lr, ch_quad.ll) < hfuzz and (vdist(line.m_internal.dir, end.lr, ch_quad.ll) < vfuzz) and (hdist(line.m_internal.dir, end.ur, ch_quad.ul) < hfuzz) and (vdist(line.m_internal.dir, end.ur, ch_quad.ul) < vfuzz):
            end.ur = ch_quad.ur
            end.lr = ch_quad.lr
            assert hits.quads[-1] == end
            return
    hits.quads.append(ch_quad)
    hits.len += 1