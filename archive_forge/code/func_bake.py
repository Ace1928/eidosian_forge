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
def bake(self, *, annots: bool=True, widgets: bool=True) -> None:
    """Convert annotations or fields to permanent content.

        Notes:
            Converts annotations or widgets to permanent page content, like
            text and vector graphics, as appropriate.
            After execution, pages will still look the same, but no longer
            have annotations, respectively no fields.
            If widgets are selected the PDF will no longer be a Form PDF.

        Args:
            annots: convert annotations
            widgets: convert form fields

        """
    pdf = _as_pdf_document(self)
    if not pdf:
        raise ValueError('not a PDF')
    mupdf.pdf_bake_document(pdf, int(annots), int(widgets))