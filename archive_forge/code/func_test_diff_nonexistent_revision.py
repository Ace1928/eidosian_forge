import os
import re
from breezy import tests, workingtree
from breezy.diff import DiffTree
from breezy.diff import format_registry as diff_format_registry
from breezy.tests import features
def test_diff_nonexistent_revision(self):
    out, err = self.run_bzr('diff -r 123', retcode=3, error_regexes=("Requested revision: '123' does not exist in branch:",))