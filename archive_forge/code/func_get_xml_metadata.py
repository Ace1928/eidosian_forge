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
def get_xml_metadata(self):
    """Get document XML metadata."""
    xml = None
    pdf = _as_pdf_document(self)
    if pdf.m_internal:
        xml = mupdf.pdf_dict_getl(mupdf.pdf_trailer(pdf), PDF_NAME('Root'), PDF_NAME('Metadata'))
    if xml and xml.m_internal:
        buff = mupdf.pdf_load_stream(xml)
        rc = JM_UnicodeFromBuffer(buff)
    else:
        rc = ''
    return rc