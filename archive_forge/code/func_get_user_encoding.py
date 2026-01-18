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
def get_user_encoding():
    """Find out what the preferred user encoding is.

    This is generally the encoding that is used for command line parameters
    and file contents. This may be different from the terminal encoding
    or the filesystem encoding.

    :return: A string defining the preferred user encoding
    """
    global _cached_user_encoding
    if _cached_user_encoding is not None:
        return _cached_user_encoding
    if os.name == 'posix' and getattr(locale, 'CODESET', None) is not None:
        user_encoding = locale.nl_langinfo(locale.CODESET)
    else:
        user_encoding = locale.getpreferredencoding(False)
    try:
        user_encoding = codecs.lookup(user_encoding).name
    except LookupError:
        if user_encoding not in ('', 'cp0'):
            sys.stderr.write('brz: warning: unknown encoding %s. Continuing with ascii encoding.\n' % user_encoding)
        user_encoding = 'ascii'
    else:
        if user_encoding == 'ascii':
            if sys.platform == 'darwin':
                user_encoding = 'utf-8'
    _cached_user_encoding = user_encoding
    return user_encoding