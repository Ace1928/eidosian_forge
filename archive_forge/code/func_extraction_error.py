import sys
import os
import io
import time
import re
import types
from typing import Protocol
import zipfile
import zipimport
import warnings
import stat
import functools
import pkgutil
import operator
import platform
import collections
import plistlib
import email.parser
import errno
import tempfile
import textwrap
import inspect
import ntpath
import posixpath
import importlib
import importlib.machinery
from pkgutil import get_importer
import _imp
from os import utime
from os import open as os_open
from os.path import isdir, split
from pkg_resources.extern.jaraco.text import (
from pkg_resources.extern import platformdirs
from pkg_resources.extern import packaging
def extraction_error(self):
    """Give an error message for problems extracting file(s)"""
    old_exc = sys.exc_info()[1]
    cache_path = self.extraction_path or get_default_cache()
    tmpl = textwrap.dedent("\n            Can't extract file(s) to egg cache\n\n            The following error occurred while trying to extract file(s)\n            to the Python egg cache:\n\n              {old_exc}\n\n            The Python egg cache directory is currently set to:\n\n              {cache_path}\n\n            Perhaps your account does not have write access to this directory?\n            You can change the cache directory by setting the PYTHON_EGG_CACHE\n            environment variable to point to an accessible directory.\n            ").lstrip()
    err = ExtractionError(tmpl.format(**locals()))
    err.manager = self
    err.cache_path = cache_path
    err.original_error = old_exc
    raise err