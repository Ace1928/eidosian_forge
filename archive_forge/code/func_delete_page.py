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
def delete_page(self, pno: int=-1):
    """ Delete one page from a PDF.
        """
    if not self.is_pdf:
        raise ValueError('is no PDF')
    if self.is_closed:
        raise ValueError('document closed')
    page_count = self.page_count
    while pno < 0:
        pno += page_count
    if pno >= page_count:
        raise ValueError('bad page number(s)')
    toc = self.get_toc()
    ol_xrefs = self.get_outline_xrefs()
    for i, item in enumerate(toc):
        if item[2] == pno + 1:
            self._remove_toc_item(ol_xrefs[i])
    self._remove_links_to(frozenset((pno,)))
    self._delete_page(pno)
    self._reset_page_refs()