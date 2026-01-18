import os
import io
import sys
import unittest
import warnings
import textwrap
from unittest import mock
from distutils.dist import Distribution, fix_help_options
from distutils.cmd import Command
from test.support import (
from test.support.os_helper import TESTFN
from distutils.tests import support
from distutils import log
def test_custom_pydistutils(self):
    if os.name == 'posix':
        user_filename = '.pydistutils.cfg'
    else:
        user_filename = 'pydistutils.cfg'
    temp_dir = self.mkdtemp()
    user_filename = os.path.join(temp_dir, user_filename)
    f = open(user_filename, 'w')
    try:
        f.write('.')
    finally:
        f.close()
    try:
        dist = Distribution()
        if sys.platform in ('linux', 'darwin'):
            os.environ['HOME'] = temp_dir
            files = dist.find_config_files()
            self.assertIn(user_filename, files)
        if sys.platform == 'win32':
            os.environ['USERPROFILE'] = temp_dir
            files = dist.find_config_files()
            self.assertIn(user_filename, files, '%r not found in %r' % (user_filename, files))
    finally:
        os.remove(user_filename)