import os
import re
import unicodedata as ud
from .. import osutils, tests
from .._termcolor import FG, color_string
from ..tests.features import UnicodeFilenameFeature
def test_basic_unknown_file(self):
    """Search for pattern in specfic file.

        If specified file is unknown, grep it anyway."""
    wd = 'foobar0'
    self.make_branch_and_tree(wd)
    os.chdir(wd)
    self._mk_unknown_file('file0.txt')
    out, err = self.run_bzr(['grep', 'line1', 'file0.txt'])
    self.assertContainsRe(out, 'file0.txt:line1', flags=TestGrep._reflags)
    self.assertEqual(len(out.splitlines()), 2)
    out, err = self.run_bzr(['grep', 'line\\d+', 'file0.txt'])
    self.assertContainsRe(out, 'file0.txt:line1', flags=TestGrep._reflags)
    self.assertEqual(len(out.splitlines()), 10)
    out, err = self.run_bzr(['grep', 'line1'])
    self.assertNotContainsRe(out, 'file0.txt', flags=TestGrep._reflags)
    self.assertEqual(len(out.splitlines()), 0)
    out, err = self.run_bzr(['grep', 'line1$'])
    self.assertNotContainsRe(out, 'file0.txt', flags=TestGrep._reflags)
    self.assertEqual(len(out.splitlines()), 0)