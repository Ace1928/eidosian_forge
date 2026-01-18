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
@classmethod
def from_mtab(cls):
    """Create a FilesystemFinder from an mtab-style file.

        Note that this will silenty ignore mtab if it doesn't exist or can not
        be opened.
        """
    try:
        return cls(read_mtab(cls.MTAB_PATH))
    except OSError as e:
        trace.mutter('Unable to read mtab: %s', e)
        return cls([])