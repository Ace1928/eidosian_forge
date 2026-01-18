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
def supports_posix_readonly():
    """Return True if 'readonly' has POSIX semantics, False otherwise.

    Notably, a win32 readonly file cannot be deleted, unlike POSIX where the
    directory controls creation/deletion, etc.

    And under win32, readonly means that the directory itself cannot be
    deleted.  The contents of a readonly directory can be changed, unlike POSIX
    where files in readonly directories cannot be added, deleted or renamed.
    """
    return sys.platform != 'win32'