import os
import re
from breezy import tests, workingtree
from breezy.diff import DiffTree
from breezy.diff import format_registry as diff_format_registry
from breezy.tests import features
def test_diff_nonexistent(self):
    self.make_example_branch()
    out, err = self.run_bzr('diff does-not-exist', retcode=3, error_regexes=('not versioned.*does-not-exist',))