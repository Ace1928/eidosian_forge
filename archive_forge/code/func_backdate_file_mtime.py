from __future__ import with_statement
from logging import getLogger
import os
import subprocess
from passlib import apache, registry
from passlib.exc import MissingBackendError
from passlib.utils.compat import irange
from passlib.tests.backports import unittest
from passlib.tests.utils import TestCase, get_file, set_file, ensure_mtime_changed
from passlib.utils.compat import u
from passlib.utils import to_bytes
from passlib.utils.handlers import to_unicode_for_identify
def backdate_file_mtime(path, offset=10):
    """backdate file's mtime by specified amount"""
    atime = os.path.getatime(path)
    mtime = os.path.getmtime(path) - offset
    os.utime(path, (atime, mtime))