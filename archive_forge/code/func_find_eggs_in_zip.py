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
def find_eggs_in_zip(importer, path_item, only=False):
    """
    Find eggs in zip files; possibly multiple nested eggs.
    """
    if importer.archive.endswith('.whl'):
        return
    metadata = EggMetadata(importer)
    if metadata.has_metadata('PKG-INFO'):
        yield Distribution.from_filename(path_item, metadata=metadata)
    if only:
        return
    for subitem in metadata.resource_listdir(''):
        if _is_egg_path(subitem):
            subpath = os.path.join(path_item, subitem)
            dists = find_eggs_in_zip(zipimport.zipimporter(subpath), subpath)
            yield from dists
        elif subitem.lower().endswith(('.dist-info', '.egg-info')):
            subpath = os.path.join(path_item, subitem)
            submeta = EggMetadata(zipimport.zipimporter(subpath))
            submeta.egg_info = subpath
            yield Distribution.from_location(path_item, subitem, submeta)