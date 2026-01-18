import os
import re
from breezy import tests, workingtree
from breezy.diff import DiffTree
from breezy.diff import format_registry as diff_format_registry
from breezy.tests import features
def test_external_diff(self):
    """Test that we can spawn an external diff process"""
    self.disable_missing_extensions_warning()
    self.make_example_branch()
    out, err = self.run_brz_subprocess('diff -Oprogress_bar=none -r 1 --diff-options -ub', universal_newlines=True, retcode=None)
    if b'Diff is not installed on this machine' in err:
        raise tests.TestSkipped("No external 'diff' is available")
    self.assertEqual(b'', err)
    self.assertStartsWith(out, b"=== added file 'goodbye'\n--- old/goodbye\t1970-01-01 00:00:00 +0000\n+++ new/goodbye\t")
    self.assertEndsWith(out, b'\n@@ -0,0 +1 @@\n+baz\n\n')