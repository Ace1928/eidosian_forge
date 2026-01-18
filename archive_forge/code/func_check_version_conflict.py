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
def check_version_conflict(self):
    if self.key == 'setuptools':
        return
    nsp = dict.fromkeys(self._get_metadata('namespace_packages.txt'))
    loc = normalize_path(self.location)
    for modname in self._get_metadata('top_level.txt'):
        if modname not in sys.modules or modname in nsp or modname in _namespace_packages:
            continue
        if modname in ('pkg_resources', 'setuptools', 'site'):
            continue
        fn = getattr(sys.modules[modname], '__file__', None)
        if fn and (normalize_path(fn).startswith(loc) or fn.startswith(self.location)):
            continue
        issue_warning('Module %s was already imported from %s, but %s is being added to sys.path' % (modname, fn, self.location))