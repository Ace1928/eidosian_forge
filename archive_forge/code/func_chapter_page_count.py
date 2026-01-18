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
def chapter_page_count(self, chapter):
    """Page count of chapter."""
    if self.is_closed:
        raise ValueError('document closed')
    chapters = mupdf.fz_count_chapters(self.this)
    if chapter < 0 or chapter >= chapters:
        raise ValueError('bad chapter number')
    pages = mupdf.fz_count_chapter_pages(self.this, chapter)
    return pages