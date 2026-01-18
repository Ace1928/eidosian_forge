import codecs
import errno
import os
import re
import stat
import sys
import time
from functools import partial
from typing import Dict, List
from .lazy_import import lazy_import
import locale
import ntpath
import posixpath
import select
import shutil
from shutil import rmtree
import socket
import subprocess
import unicodedata
from breezy import (
from breezy.i18n import gettext
from hashlib import md5
from hashlib import sha1 as sha
import breezy
from . import errors
def quotefn(f):
    """Return a quoted filename filename

    This previously used backslash quoting, but that works poorly on
    Windows."""
    global _QUOTE_RE
    if _QUOTE_RE is None:
        _QUOTE_RE = re.compile('([^a-zA-Z0-9.,:/\\\\_~-])')
    if _QUOTE_RE.search(f):
        return '"' + f + '"'
    else:
        return f