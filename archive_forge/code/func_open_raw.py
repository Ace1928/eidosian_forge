from __future__ import absolute_import, unicode_literals
import io
import posixpath
import sys
from os import environ
from pybtex.exceptions import PybtexError
from pybtex.kpathsea import kpsewhich
def open_raw(filename, mode='rb', encoding=None):
    return _open(io.open, filename, mode)