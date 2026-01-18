import unittest
import os
import sys
import tarfile
from os.path import splitdrive
import warnings
from distutils import archive_util
from distutils.archive_util import (check_archive_formats, make_tarball,
from distutils.spawn import find_executable, spawn
from distutils.tests import support
from test.support import patch
from test.support.os_helper import change_cwd
from test.support.warnings_helper import check_warnings
def fake_zipfile(*a, **kw):
    if kw.get('compression', None) == zipfile.ZIP_STORED:
        called.append((a, kw))
    return zipfile_class(*a, **kw)