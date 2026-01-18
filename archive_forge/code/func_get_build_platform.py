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
def get_build_platform():
    """Return this platform's string for platform-specific distributions

    XXX Currently this is the same as ``distutils.util.get_platform()``, but it
    needs some hacks for Linux and macOS.
    """
    from sysconfig import get_platform
    plat = get_platform()
    if sys.platform == 'darwin' and (not plat.startswith('macosx-')):
        try:
            version = _macos_vers()
            machine = os.uname()[4].replace(' ', '_')
            return 'macosx-%d.%d-%s' % (int(version[0]), int(version[1]), _macos_arch(machine))
        except ValueError:
            pass
    return plat