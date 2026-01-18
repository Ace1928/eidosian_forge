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
def file_ns_handler(importer, path_item, packageName, module):
    """Compute an ns-package subpath for a filesystem or zipfile importer"""
    subpath = os.path.join(path_item, packageName.split('.')[-1])
    normalized = _normalize_cached(subpath)
    for item in module.__path__:
        if _normalize_cached(item) == normalized:
            break
    else:
        return subpath