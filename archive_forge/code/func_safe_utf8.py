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
def safe_utf8(unicode_or_utf8_string):
    """Coerce unicode_or_utf8_string to a utf8 string.

    If it is a str, it is returned.
    If it is Unicode, it is encoded into a utf-8 string.
    """
    if isinstance(unicode_or_utf8_string, bytes):
        try:
            unicode_or_utf8_string.decode('utf-8')
        except UnicodeDecodeError:
            raise errors.BzrBadParameterNotUnicode(unicode_or_utf8_string)
        return unicode_or_utf8_string
    return unicode_or_utf8_string.encode('utf-8')