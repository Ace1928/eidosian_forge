from __future__ import absolute_import
import cython
import os
import sys
import re
import io
import codecs
import glob
import shutil
import tempfile
from functools import wraps
from . import __version__ as cython_version
def open_source_from_loader(loader, source_filename, encoding=None, error_handling=None):
    nrmpath = os.path.normpath(source_filename)
    arcname = nrmpath[len(loader.archive) + 1:]
    data = loader.get_data(arcname)
    return io.TextIOWrapper(io.BytesIO(data), encoding=encoding, errors=error_handling)