import os
import re
from breezy import tests, workingtree
from breezy.diff import DiffTree
from breezy.diff import format_registry as diff_format_registry
from breezy.tests import features
def test_diff_illegal_prefix_value(self):
    out, err = self.run_bzr('diff --prefix old/', retcode=3)
    self.assertContainsRe(err, '--prefix expects two values separated by a colon')