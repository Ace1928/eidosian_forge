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
@property
def parsed_version(self):
    if not hasattr(self, '_parsed_version'):
        try:
            self._parsed_version = parse_version(self.version)
        except packaging.version.InvalidVersion as ex:
            info = f'(package: {self.project_name})'
            if hasattr(ex, 'add_note'):
                ex.add_note(info)
                raise
            raise packaging.version.InvalidVersion(f'{str(ex)} {info}') from None
    return self._parsed_version