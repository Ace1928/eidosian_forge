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
def resolve_link(self, uri=None, chapters=0):
    """Calculate internal link destination.

        Args:
            uri: (str) some Link.uri
            chapters: (bool) whether to use (chapter, page) format
        Returns:
            (page_id, x, y) where x, y are point coordinates on the page.
            page_id is either page number (if chapters=0), or (chapter, pno).
        """
    if not uri:
        if chapters:
            return ((-1, -1), 0, 0)
        return (-1, 0, 0)
    try:
        loc, xp, yp = mupdf.fz_resolve_link(self.this, uri)
    except Exception:
        if g_exceptions_verbose:
            exception_info()
        if chapters:
            return ((-1, -1), 0, 0)
        return (-1, 0, 0)
    if chapters:
        return ((loc.chapter, loc.page), xp, yp)
    pno = mupdf.fz_page_number_from_location(self.this, loc)
    return (pno, xp, yp)