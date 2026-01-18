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
def annots(self, types=None):
    """ Generator over the annotations of a page.

        Args:
            types: (list) annotation types to subselect from. If none,
                   all annotations are returned. E.g. types=[PDF_ANNOT_LINE]
                   will only yield line annotations.
        """
    skip_types = (mupdf.PDF_ANNOT_LINK, mupdf.PDF_ANNOT_POPUP, mupdf.PDF_ANNOT_WIDGET)
    if not hasattr(types, '__getitem__'):
        annot_xrefs = [a[0] for a in self.annot_xrefs() if a[1] not in skip_types]
    else:
        annot_xrefs = [a[0] for a in self.annot_xrefs() if a[1] in types and a[1] not in skip_types]
    for xref in annot_xrefs:
        annot = self.load_annot(xref)
        annot._yielded = True
        yield annot