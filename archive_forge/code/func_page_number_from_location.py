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
def page_number_from_location(self, page_id):
    """Convert (chapter, pno) to page number."""
    if type(page_id) is int:
        np = self.page_count
        while page_id < 0:
            page_id += np
        page_id = (0, page_id)
    if page_id not in self:
        raise ValueError('page id not in document')
    chapter, pno = page_id
    loc = mupdf.fz_make_location(chapter, pno)
    page_n = mupdf.fz_page_number_from_location(self.this, loc)
    return page_n