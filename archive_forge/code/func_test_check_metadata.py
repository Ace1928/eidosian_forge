import os
import textwrap
import unittest
from distutils.command.check import check, HAS_DOCUTILS
from distutils.tests import support
from distutils.errors import DistutilsSetupError
def test_check_metadata(self):
    cmd = self._run()
    self.assertEqual(cmd._warnings, 2)
    metadata = {'url': 'xxx', 'author': 'xxx', 'author_email': 'xxx', 'name': 'xxx', 'version': 'xxx'}
    cmd = self._run(metadata)
    self.assertEqual(cmd._warnings, 0)
    self.assertRaises(DistutilsSetupError, self._run, {}, **{'strict': 1})
    cmd = self._run(metadata, strict=1)
    self.assertEqual(cmd._warnings, 0)
    metadata = {'url': 'xxx', 'author': 'Éric', 'author_email': 'xxx', 'name': 'xxx', 'version': 'xxx', 'description': 'Something about esszet ß', 'long_description': 'More things about esszet ß'}
    cmd = self._run(metadata)
    self.assertEqual(cmd._warnings, 0)