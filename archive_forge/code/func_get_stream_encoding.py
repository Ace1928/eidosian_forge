from __future__ import absolute_import, unicode_literals
import io
import posixpath
import sys
from os import environ
from pybtex.exceptions import PybtexError
from pybtex.kpathsea import kpsewhich
def get_stream_encoding(stream):
    stream_encoding = getattr(stream, 'encoding', None)
    return stream_encoding or get_default_encoding()