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
def position_in_sys_path(path):
    """
        Return the ordinal of the path based on its position in sys.path
        """
    path_parts = path.split(os.sep)
    module_parts = package_name.count('.') + 1
    parts = path_parts[:-module_parts]
    return safe_sys_path_index(_normalize_cached(os.sep.join(parts)))