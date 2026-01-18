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
def file_kind_from_stat_mode_thunk(mode):
    global file_kind_from_stat_mode
    if file_kind_from_stat_mode is file_kind_from_stat_mode_thunk:
        try:
            from ._readdir_pyx import UTF8DirReader
            file_kind_from_stat_mode = UTF8DirReader().kind_from_mode
        except ImportError:
            from ._readdir_py import _kind_from_mode as file_kind_from_stat_mode
    return file_kind_from_stat_mode(mode)