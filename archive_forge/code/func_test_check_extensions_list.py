import sys
import os
from io import StringIO
import textwrap
from distutils.core import Distribution
from distutils.command.build_ext import build_ext
from distutils import sysconfig
from distutils.tests.support import (TempdirManager, LoggingSilencer,
from distutils.extension import Extension
from distutils.errors import (
import unittest
from test import support
from test.support import os_helper
from test.support.script_helper import assert_python_ok
from test.support import threading_helper
def test_check_extensions_list(self):
    dist = Distribution()
    cmd = self.build_ext(dist)
    cmd.finalize_options()
    self.assertRaises(DistutilsSetupError, cmd.check_extensions_list, 'foo')
    exts = [('bar', 'foo', 'bar'), 'foo']
    self.assertRaises(DistutilsSetupError, cmd.check_extensions_list, exts)
    exts = [('foo-bar', '')]
    self.assertRaises(DistutilsSetupError, cmd.check_extensions_list, exts)
    exts = [('foo.bar', '')]
    self.assertRaises(DistutilsSetupError, cmd.check_extensions_list, exts)
    exts = [('foo.bar', {'sources': [''], 'libraries': 'foo', 'some': 'bar'})]
    cmd.check_extensions_list(exts)
    ext = exts[0]
    self.assertIsInstance(ext, Extension)
    self.assertEqual(ext.libraries, 'foo')
    self.assertFalse(hasattr(ext, 'some'))
    exts = [('foo.bar', {'sources': [''], 'libraries': 'foo', 'some': 'bar', 'macros': [('1', '2', '3'), 'foo']})]
    self.assertRaises(DistutilsSetupError, cmd.check_extensions_list, exts)
    exts[0][1]['macros'] = [('1', '2'), ('3',)]
    cmd.check_extensions_list(exts)
    self.assertEqual(exts[0].undef_macros, ['3'])
    self.assertEqual(exts[0].define_macros, [('1', '2')])