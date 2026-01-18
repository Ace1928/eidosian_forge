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
def safe_unicode(unicode_or_utf8_string):
    """Coerce unicode_or_utf8_string into unicode.

    If it is unicode, it is returned.
    Otherwise it is decoded from utf-8. If decoding fails, the exception is
    wrapped in a BzrBadParameterNotUnicode exception.
    """
    if isinstance(unicode_or_utf8_string, str):
        return unicode_or_utf8_string
    try:
        return unicode_or_utf8_string.decode('utf8')
    except UnicodeDecodeError:
        raise errors.BzrBadParameterNotUnicode(unicode_or_utf8_string)