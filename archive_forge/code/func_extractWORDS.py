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
def extractWORDS(self, delimiters=None):
    """Return a list with text word information."""
    if g_use_extra:
        return extra.extractWORDS(self.this, delimiters)
    buflen = 0
    block_n = -1
    wbbox = mupdf.FzRect(mupdf.FzRect.Fixed_EMPTY)
    this_tpage = self.this
    tp_rect = mupdf.FzRect(this_tpage.m_internal.mediabox)
    lines = None
    buff = mupdf.fz_new_buffer(64)
    lines = []
    for block in this_tpage:
        block_n += 1
        if block.m_internal.type != mupdf.FZ_STEXT_BLOCK_TEXT:
            continue
        line_n = -1
        for line in block:
            line_n += 1
            word_n = 0
            mupdf.fz_clear_buffer(buff)
            buflen = 0
            for ch in line:
                cbbox = JM_char_bbox(line, ch)
                if not JM_rects_overlap(tp_rect, cbbox) and (not mupdf.fz_is_infinite_rect(tp_rect)):
                    continue
                word_delimiter = JM_is_word_delimiter(ch.m_internal.c, delimiters)
                if word_delimiter:
                    if buflen == 0:
                        continue
                    if not mupdf.fz_is_empty_rect(wbbox):
                        word_n, wbbox = JM_append_word(lines, buff, wbbox, block_n, line_n, word_n)
                    mupdf.fz_clear_buffer(buff)
                    buflen = 0
                    continue
                JM_append_rune(buff, ch.m_internal.c)
                buflen += 1
                wbbox = mupdf.fz_union_rect(wbbox, JM_char_bbox(line, ch))
            if buflen and (not mupdf.fz_is_empty_rect(wbbox)):
                word_n, wbbox = JM_append_word(lines, buff, wbbox, block_n, line_n, word_n)
            buflen = 0
    return lines