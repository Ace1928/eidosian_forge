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
def dist_factory(path_item, entry, only):
    """Return a dist_factory for the given entry."""
    lower = entry.lower()
    is_egg_info = lower.endswith('.egg-info')
    is_dist_info = lower.endswith('.dist-info') and os.path.isdir(os.path.join(path_item, entry))
    is_meta = is_egg_info or is_dist_info
    return distributions_from_metadata if is_meta else find_distributions if not only and _is_egg_path(entry) else resolve_egg_link if not only and lower.endswith('.egg-link') else NoDists()