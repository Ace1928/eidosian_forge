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
def declare_namespace(packageName):
    """Declare that package 'packageName' is a namespace package"""
    msg = f'Deprecated call to `pkg_resources.declare_namespace({packageName!r})`.\nImplementing implicit namespace packages (as specified in PEP 420) is preferred to `pkg_resources.declare_namespace`. See https://setuptools.pypa.io/en/latest/references/keywords.html#keyword-namespace-packages'
    warnings.warn(msg, DeprecationWarning, stacklevel=2)
    _imp.acquire_lock()
    try:
        if packageName in _namespace_packages:
            return
        path = sys.path
        parent, _, _ = packageName.rpartition('.')
        if parent:
            declare_namespace(parent)
            if parent not in _namespace_packages:
                __import__(parent)
            try:
                path = sys.modules[parent].__path__
            except AttributeError as e:
                raise TypeError('Not a package:', parent) from e
        _namespace_packages.setdefault(parent or None, []).append(packageName)
        _namespace_packages.setdefault(packageName, [])
        for path_item in path:
            _handle_ns(packageName, path_item)
    finally:
        _imp.release_lock()