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
def add_redact_annot(self, quad, text: OptStr=None, fontname: OptStr=None, fontsize: float=11, align: int=0, fill: OptSeq=None, text_color: OptSeq=None, cross_out: bool=True) -> Annot:
    """Add a 'Redact' annotation."""
    da_str = None
    if text:
        CheckColor(fill)
        CheckColor(text_color)
        if not fontname:
            fontname = 'Helv'
        if not fontsize:
            fontsize = 11
        if not text_color:
            text_color = (0, 0, 0)
        if hasattr(text_color, '__float__'):
            text_color = (text_color, text_color, text_color)
        if len(text_color) > 3:
            text_color = text_color[:3]
        fmt = '{:g} {:g} {:g} rg /{f:s} {s:g} Tf'
        da_str = fmt.format(*text_color, f=fontname, s=fontsize)
        if fill is None:
            fill = (1, 1, 1)
        if fill:
            if hasattr(fill, '__float__'):
                fill = (fill, fill, fill)
            if len(fill) > 3:
                fill = fill[:3]
    old_rotation = annot_preprocess(self)
    try:
        annot = self._add_redact_annot(quad, text=text, da_str=da_str, align=align, fill=fill)
    finally:
        if old_rotation != 0:
            self.set_rotation(old_rotation)
    annot_postprocess(self, annot)
    if cross_out:
        ap_tab = annot._getAP().splitlines()[:-1]
        _, LL, LR, UR, UL = ap_tab
        ap_tab.append(LR)
        ap_tab.append(LL)
        ap_tab.append(UR)
        ap_tab.append(LL)
        ap_tab.append(UL)
        ap_tab.append(b'S')
        ap = b'\n'.join(ap_tab)
        annot._setAP(ap, 0)
    return annot