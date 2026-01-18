import contextlib
import os
import re
import subprocess
import sys
import tempfile
from io import BytesIO
from .. import diff, errors, osutils
from .. import revision as _mod_revision
from .. import revisionspec, revisiontree, tests
from ..tests import EncodingAdapter, features
from ..tests.scenarios import load_tests_apply_scenarios
def test_diff_remove_files(self):
    tree1 = self.b.repository.revision_tree(b'rev-3')
    tree2 = self.b.repository.revision_tree(b'rev-4')
    output = get_diff_as_string(tree1, tree2)
    self.assertEqualDiff(output, b"=== removed file 'file2'\n--- old/file2\t2006-04-03 00:00:00 +0000\n+++ new/file2\t1970-01-01 00:00:00 +0000\n@@ -1,1 +0,0 @@\n-file2 contents at rev 3\n\n")