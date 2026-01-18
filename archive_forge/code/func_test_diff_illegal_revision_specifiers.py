import os
import re
from breezy import tests, workingtree
from breezy.diff import DiffTree
from breezy.diff import format_registry as diff_format_registry
from breezy.tests import features
def test_diff_illegal_revision_specifiers(self):
    out, err = self.run_bzr('diff -r 1..23..123', retcode=3, error_regexes=('one or two revision specifiers',))