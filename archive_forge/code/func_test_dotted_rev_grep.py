import os
import re
import unicodedata as ud
from .. import osutils, tests
from .._termcolor import FG, color_string
from ..tests.features import UnicodeFilenameFeature
def test_dotted_rev_grep(self):
    """Grep in dotted revs
        """
    wd0 = 'foobar0'
    wd1 = 'foobar1'
    self.make_branch_and_tree(wd0)
    os.chdir(wd0)
    self._mk_versioned_file('file0.txt')
    os.chdir('..')
    out, err = self.run_bzr(['branch', wd0, wd1])
    os.chdir(wd1)
    self._mk_versioned_file('file1.txt')
    self._update_file('file1.txt', 'text 0\n')
    self._update_file('file1.txt', 'text 1\n')
    self._update_file('file1.txt', 'text 2\n')
    os.chdir(osutils.pathjoin('..', wd0))
    out, err = self.run_bzr(['merge', osutils.pathjoin('..', wd1)])
    out, err = self.run_bzr(['ci', '-m', 'merged'])
    out, err = self.run_bzr(['grep', '-r', '1.1.1..1.1.4', 'text'])
    self.assertContainsRe(out, 'file1.txt~1.1.2:text 0', flags=TestGrep._reflags)
    self.assertContainsRe(out, 'file1.txt~1.1.3:text 1', flags=TestGrep._reflags)
    self.assertContainsRe(out, 'file1.txt~1.1.3:text 1', flags=TestGrep._reflags)
    self.assertContainsRe(out, 'file1.txt~1.1.4:text 0', flags=TestGrep._reflags)
    self.assertContainsRe(out, 'file1.txt~1.1.4:text 1', flags=TestGrep._reflags)
    self.assertContainsRe(out, 'file1.txt~1.1.4:text 2', flags=TestGrep._reflags)
    self.assertEqual(len(out.splitlines()), 6)