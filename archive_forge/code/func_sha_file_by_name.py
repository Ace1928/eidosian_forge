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
def sha_file_by_name(fname):
    """Calculate the SHA1 of a file by reading the full text"""
    s = sha()
    f = os.open(fname, os.O_RDONLY | O_BINARY | O_NOINHERIT)
    try:
        while True:
            b = os.read(f, 1 << 16)
            if not b:
                return _hexdigest(s)
            s.update(b)
    finally:
        os.close(f)