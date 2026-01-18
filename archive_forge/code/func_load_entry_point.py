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
def load_entry_point(self, group, name):
    """Return the `name` entry point of `group` or raise ImportError"""
    ep = self.get_entry_info(group, name)
    if ep is None:
        raise ImportError('Entry point %r not found' % ((group, name),))
    return ep.load()