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
def distributions_from_metadata(path):
    root = os.path.dirname(path)
    if os.path.isdir(path):
        if len(os.listdir(path)) == 0:
            return
        metadata = PathMetadata(root, path)
    else:
        metadata = FileMetadata(path)
    entry = os.path.basename(path)
    yield Distribution.from_location(root, entry, metadata, precedence=DEVELOP_DIST)