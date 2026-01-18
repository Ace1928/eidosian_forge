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
def copy_ownership_from_path(dst, src=None):
    """Copy usr/grp ownership from src file/dir to dst file/dir.

    If src is None, the containing directory is used as source. If chown
    fails, the error is ignored and a warning is printed.
    """
    chown = getattr(os, 'chown', None)
    if chown is None:
        return
    if src is None:
        src = os.path.dirname(dst)
        if src == '':
            src = '.'
    try:
        s = os.stat(src)
        chown(dst, s.st_uid, s.st_gid)
    except OSError:
        trace.warning('Unable to copy ownership from "%s" to "%s". You may want to set it manually.', src, dst)
        trace.log_exception_quietly()