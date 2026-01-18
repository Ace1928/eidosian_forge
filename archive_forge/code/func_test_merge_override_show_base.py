import doctest
import os
from testtools import matchers
from breezy import (branch, controldir, merge_directive, osutils, tests,
from breezy.bzr import conflicts
from breezy.tests import scenarios, script
def test_merge_override_show_base(self):
    tree, other = self.create_conflicting_branches()
    self.run_bzr('merge ../other --show-base', working_dir='tree', retcode=1)
    self.assertEqualDiff(b'a\n<<<<<<< TREE\nB\nC\n||||||| BASE-REVISION\nb\nc\n=======\nB\nD\n>>>>>>> MERGE-SOURCE\n', tree.get_file_text('fname'))