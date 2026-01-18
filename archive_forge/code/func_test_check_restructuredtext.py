import os
import textwrap
import unittest
from distutils.command.check import check, HAS_DOCUTILS
from distutils.tests import support
from distutils.errors import DistutilsSetupError
@unittest.skipUnless(HAS_DOCUTILS, "won't test without docutils")
def test_check_restructuredtext(self):
    broken_rest = 'title\n===\n\ntest'
    pkg_info, dist = self.create_dist(long_description=broken_rest)
    cmd = check(dist)
    cmd.check_restructuredtext()
    self.assertEqual(cmd._warnings, 1)
    metadata = {'url': 'xxx', 'author': 'xxx', 'author_email': 'xxx', 'name': 'xxx', 'version': 'xxx', 'long_description': broken_rest}
    self.assertRaises(DistutilsSetupError, self._run, metadata, **{'strict': 1, 'restructuredtext': 1})
    metadata['long_description'] = 'title\n=====\n\ntest ß'
    cmd = self._run(metadata, strict=1, restructuredtext=1)
    self.assertEqual(cmd._warnings, 0)
    metadata['long_description'] = 'title\n=====\n\n.. include:: includetest.rst'
    cmd = self._run(metadata, cwd=HERE, strict=1, restructuredtext=1)
    self.assertEqual(cmd._warnings, 0)