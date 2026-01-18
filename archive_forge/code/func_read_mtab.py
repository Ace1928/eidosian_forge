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
def read_mtab(path):
    """Read an fstab-style file and extract mountpoint+filesystem information.

    :param path: Path to read from
    :yield: Tuples with mountpoints (as bytestrings) and filesystem names
    """
    with open(path, 'rb') as f:
        for line in f:
            if line.startswith(b'#'):
                continue
            cols = line.split()
            if len(cols) < 3:
                continue
            yield (cols[1], cols[2].decode('ascii', 'replace'))