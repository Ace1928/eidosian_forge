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
def annot_names(self):
    """
        page get list of annot names
        """
    'List of names of annotations, fields and links.'
    CheckParent(self)
    page = self._pdf_page()
    if not page.m_internal:
        return []
    return JM_get_annot_id_list(page)