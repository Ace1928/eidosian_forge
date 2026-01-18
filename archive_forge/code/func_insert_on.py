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
def insert_on(self, path, loc=None, replace=False):
    """Ensure self.location is on path

        If replace=False (default):
            - If location is already in path anywhere, do nothing.
            - Else:
              - If it's an egg and its parent directory is on path,
                insert just ahead of the parent.
              - Else: add to the end of path.
        If replace=True:
            - If location is already on path anywhere (not eggs)
              or higher priority than its parent (eggs)
              do nothing.
            - Else:
              - If it's an egg and its parent directory is on path,
                insert just ahead of the parent,
                removing any lower-priority entries.
              - Else: add it to the front of path.
        """
    loc = loc or self.location
    if not loc:
        return
    nloc = _normalize_cached(loc)
    bdir = os.path.dirname(nloc)
    npath = [p and _normalize_cached(p) or p for p in path]
    for p, item in enumerate(npath):
        if item == nloc:
            if replace:
                break
            else:
                return
        elif item == bdir and self.precedence == EGG_DIST:
            if not replace and nloc in npath[p:]:
                return
            if path is sys.path:
                self.check_version_conflict()
            path.insert(p, loc)
            npath.insert(p, nloc)
            break
    else:
        if path is sys.path:
            self.check_version_conflict()
        if replace:
            path.insert(0, loc)
        else:
            path.append(loc)
        return
    while True:
        try:
            np = npath.index(nloc, p + 1)
        except ValueError:
            break
        else:
            del npath[np], path[np]
            p = np
    return