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
def jm_lineart_fill_text(dev, ctx, text, ctm, colorspace, color, alpha, color_params):
    if 0:
        log(f'type(ctx)={type(ctx)!r} ctx={ctx!r}')
        log(f'type(dev)={type(dev)!r} dev={dev!r}')
        log(f'type(text)={type(text)!r} text={text!r}')
        log(f'type(ctm)={type(ctm)!r} ctm={ctm!r}')
        log(f'type(colorspace)={type(colorspace)!r} colorspace={colorspace!r}')
        log(f'type(color)={type(color)!r} color={color!r}')
        log(f'type(alpha)={type(alpha)!r} alpha={alpha!r}')
        log(f'type(color_params)={type(color_params)!r} color_params={color_params!r}')
    jm_trace_text(dev, text, 0, ctm, colorspace, color, alpha, dev.seqno)
    dev.seqno += 1