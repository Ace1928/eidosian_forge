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
def jm_bbox_fill_image(dev, ctx, image, ctm, alpha, color_params):
    r = mupdf.FzRect(mupdf.FzRect.Fixed_UNIT)
    r = mupdf.ll_fz_transform_rect(r.internal(), ctm)
    jm_bbox_add_rect(dev, ctx, r, 'fill-image')