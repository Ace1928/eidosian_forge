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
def test_empty_options(self):
    warns = []

    def _warn(msg):
        warns.append(msg)
    self.addCleanup(setattr, warnings, 'warn', warnings.warn)
    warnings.warn = _warn
    dist = Distribution(attrs={'author': 'xxx', 'name': 'xxx', 'version': 'xxx', 'url': 'xxxx', 'options': {}})
    self.assertEqual(len(warns), 0)
    self.assertNotIn('options', dir(dist))