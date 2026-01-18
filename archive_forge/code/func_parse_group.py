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
@classmethod
def parse_group(cls, group, lines, dist=None):
    """Parse an entry point group"""
    if not MODULE(group):
        raise ValueError('Invalid group name', group)
    this = {}
    for line in yield_lines(lines):
        ep = cls.parse(line, dist)
        if ep.name in this:
            raise ValueError('Duplicate entry point', group, ep.name)
        this[ep.name] = ep
    return this