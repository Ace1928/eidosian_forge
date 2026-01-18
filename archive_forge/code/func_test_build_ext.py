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
@support.requires_subprocess()
def test_build_ext(self):
    cmd = support.missing_compiler_executable()
    if cmd is not None:
        self.skipTest('The %r command is not found' % cmd)
    global ALREADY_TESTED
    copy_xxmodule_c(self.tmp_dir)
    xx_c = os.path.join(self.tmp_dir, 'xxmodule.c')
    xx_ext = Extension('xx', [xx_c])
    dist = Distribution({'name': 'xx', 'ext_modules': [xx_ext]})
    dist.package_dir = self.tmp_dir
    cmd = self.build_ext(dist)
    fixup_build_ext(cmd)
    cmd.build_lib = self.tmp_dir
    cmd.build_temp = self.tmp_dir
    old_stdout = sys.stdout
    if not support.verbose:
        sys.stdout = StringIO()
    try:
        cmd.ensure_finalized()
        cmd.run()
    finally:
        sys.stdout = old_stdout
    if ALREADY_TESTED:
        self.skipTest('Already tested in %s' % ALREADY_TESTED)
    else:
        ALREADY_TESTED = type(self).__name__
    code = textwrap.dedent(f"\n            tmp_dir = {self.tmp_dir!r}\n\n            import sys\n            import unittest\n            from test import support\n\n            sys.path.insert(0, tmp_dir)\n            import xx\n\n            class Tests(unittest.TestCase):\n                def test_xx(self):\n                    for attr in ('error', 'foo', 'new', 'roj'):\n                        self.assertTrue(hasattr(xx, attr))\n\n                    self.assertEqual(xx.foo(2, 5), 7)\n                    self.assertEqual(xx.foo(13,15), 28)\n                    self.assertEqual(xx.new().demo(), None)\n                    if support.HAVE_DOCSTRINGS:\n                        doc = 'This is a template module just for instruction.'\n                        self.assertEqual(xx.__doc__, doc)\n                    self.assertIsInstance(xx.Null(), xx.Null)\n                    self.assertIsInstance(xx.Str(), xx.Str)\n\n\n            unittest.main()\n        ")
    assert_python_ok('-c', code)