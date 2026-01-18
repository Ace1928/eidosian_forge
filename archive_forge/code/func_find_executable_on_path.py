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
def find_executable_on_path(name):
    """Finds an executable on the PATH.

    On Windows, this will try to append each extension in the PATHEXT
    environment variable to the name, if it cannot be found with the name
    as given.

    :param name: The base name of the executable.
    :return: The path to the executable found or None.
    """
    if sys.platform == 'win32':
        exts = os.environ.get('PATHEXT', '').split(os.pathsep)
        exts = [ext.lower() for ext in exts]
        base, ext = os.path.splitext(name)
        if ext != '':
            if ext.lower() not in exts:
                return None
            name = base
            exts = [ext]
    else:
        exts = ['']
    path = os.environ.get('PATH')
    if path is not None:
        path = path.split(os.pathsep)
        for ext in exts:
            for d in path:
                f = os.path.join(d, name) + ext
                if os.access(f, os.X_OK):
                    return f
    if sys.platform == 'win32':
        app_path = win32utils.get_app_path(name)
        if app_path != name:
            return app_path
    return None