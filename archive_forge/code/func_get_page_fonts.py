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
def get_page_fonts(self, pno: int, full: bool=False) -> list:
    """Retrieve a list of fonts used on a page.
        """
    if self.is_closed or self.is_encrypted:
        raise ValueError('document closed or encrypted')
    if not self.is_pdf:
        return ()
    if type(pno) is not int:
        try:
            pno = pno.number
        except Exception:
            exception_info()
            raise ValueError('need a Page or page number')
    val = self._getPageInfo(pno, 1)
    if full is False:
        return [v[:-1] for v in val]
    return val