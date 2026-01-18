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
def strip_outlines(doc, outlines, page_count, page_object_nums, names_list):
    if not outlines.m_internal:
        return 0
    first = mupdf.pdf_dict_get(outlines, PDF_NAME('First'))
    if not first.m_internal:
        nc = 0
    else:
        nc, first, last = strip_outline(doc, first, page_count, page_object_nums, names_list)
    if nc == 0:
        mupdf.pdf_dict_del(outlines, PDF_NAME('First'))
        mupdf.pdf_dict_del(outlines, PDF_NAME('Last'))
        mupdf.pdf_dict_del(outlines, PDF_NAME('Count'))
    else:
        old_count = mupdf.pdf_to_int(mupdf.pdf_dict_get(outlines, PDF_NAME('Count')))
        mupdf.pdf_dict_put(outlines, PDF_NAME('First'), first)
        mupdf.pdf_dict_put(outlines, PDF_NAME('Last'), last)
        mupdf.pdf_dict_put(outlines, PDF_NAME('Count'), mupdf.pdf_new_int(nc if old_count > 0 else -nc))
    return nc