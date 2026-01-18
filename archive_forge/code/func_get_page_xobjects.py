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
def get_page_xobjects(self, pno: int) -> list:
    """Retrieve a list of XObjects used on a page.
        """
    if self.is_closed or self.is_encrypted:
        raise ValueError('document closed or encrypted')
    if not self.is_pdf:
        return ()
    val = self._getPageInfo(pno, 3)
    return val