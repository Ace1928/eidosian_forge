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
def get_highlight_selection(page, start: point_like=None, stop: point_like=None, clip: rect_like=None) -> list:
    """Return rectangles of text lines between two points.

    Notes:
        The default of 'start' is top-left of 'clip'. The default of 'stop'
        is bottom-reight of 'clip'.

    Args:
        start: start point_like
        stop: end point_like, must be 'below' start
        clip: consider this rect_like only, default is page rectangle
    Returns:
        List of line bbox intersections with the area established by the
        parameters.
    """
    if clip is None:
        clip = page.rect
    clip = Rect(clip)
    if start is None:
        start = clip.tl
    if stop is None:
        stop = clip.br
    clip.y0 = start.y
    clip.y1 = stop.y
    if clip.is_empty or clip.is_infinite:
        return []
    blocks = page.get_text('dict', flags=0, clip=clip)['blocks']
    lines = []
    for b in blocks:
        bbox = Rect(b['bbox'])
        if bbox.is_infinite or bbox.is_empty:
            continue
        for line in b['lines']:
            bbox = Rect(line['bbox'])
            if bbox.is_infinite or bbox.is_empty:
                continue
            lines.append(bbox)
    if lines == []:
        return lines
    lines.sort(key=lambda bbox: bbox.y1)
    bboxf = lines.pop(0)
    if bboxf.y0 - start.y <= 0.1 * bboxf.height:
        r = Rect(start.x, bboxf.y0, bboxf.br)
        if not (r.is_empty or r.is_infinite):
            lines.insert(0, r)
    else:
        lines.insert(0, bboxf)
    if lines == []:
        return lines
    bboxl = lines.pop()
    if stop.y - bboxl.y1 <= 0.1 * bboxl.height:
        r = Rect(bboxl.tl, stop.x, bboxl.y1)
        if not (r.is_empty or r.is_infinite):
            lines.append(r)
    else:
        lines.append(bboxl)
    return lines