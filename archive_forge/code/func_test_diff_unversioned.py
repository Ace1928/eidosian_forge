import os
import re
from breezy import tests, workingtree
from breezy.diff import DiffTree
from breezy.diff import format_registry as diff_format_registry
from breezy.tests import features
def test_diff_unversioned(self):
    self.make_example_branch()
    self.build_tree(['unversioned-file'])
    out, err = self.run_bzr('diff unversioned-file', retcode=3)
    self.assertContainsRe(err, 'not versioned.*unversioned-file')