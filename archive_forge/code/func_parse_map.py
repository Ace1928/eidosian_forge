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
def parse_map(cls, data, dist=None):
    """Parse a map of entry point groups"""
    if isinstance(data, dict):
        data = data.items()
    else:
        data = split_sections(data)
    maps = {}
    for group, lines in data:
        if group is None:
            if not lines:
                continue
            raise ValueError('Entry points must be listed in groups')
        group = group.strip()
        if group in maps:
            raise ValueError('Duplicate group name', group)
        maps[group] = cls.parse_group(group, lines, dist)
    return maps