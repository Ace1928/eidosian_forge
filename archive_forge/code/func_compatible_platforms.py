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
def compatible_platforms(provided, required):
    """Can code for the `provided` platform run on the `required` platform?

    Returns true if either platform is ``None``, or the platforms are equal.

    XXX Needs compatibility checks for Linux and other unixy OSes.
    """
    if provided is None or required is None or provided == required:
        return True
    reqMac = macosVersionString.match(required)
    if reqMac:
        provMac = macosVersionString.match(provided)
        if not provMac:
            provDarwin = darwinVersionString.match(provided)
            if provDarwin:
                dversion = int(provDarwin.group(1))
                macosversion = '%s.%s' % (reqMac.group(1), reqMac.group(2))
                if dversion == 7 and macosversion >= '10.3' or (dversion == 8 and macosversion >= '10.4'):
                    return True
            return False
        if provMac.group(1) != reqMac.group(1) or provMac.group(3) != reqMac.group(3):
            return False
        if int(provMac.group(2)) > int(reqMac.group(2)):
            return False
        return True
    return False