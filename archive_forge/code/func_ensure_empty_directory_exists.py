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
def ensure_empty_directory_exists(path, exception_class):
    """Make sure a local directory exists and is empty.

    If it does not exist, it is created.  If it exists and is not empty, an
    instance of exception_class is raised.
    """
    try:
        os.mkdir(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
        if os.listdir(path) != []:
            raise exception_class(path)