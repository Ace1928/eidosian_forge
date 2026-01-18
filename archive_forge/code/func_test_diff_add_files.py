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
def test_diff_add_files(self):
    tree1 = self.b.repository.revision_tree(_mod_revision.NULL_REVISION)
    tree2 = self.b.repository.revision_tree(b'rev-1')
    output = get_diff_as_string(tree1, tree2)
    self.assertEqualDiff(output, b"=== added file 'file1'\n--- old/file1\t1970-01-01 00:00:00 +0000\n+++ new/file1\t2006-04-01 00:00:00 +0000\n@@ -0,0 +1,1 @@\n+file1 contents at rev 1\n\n=== added file 'file2'\n--- old/file2\t1970-01-01 00:00:00 +0000\n+++ new/file2\t2006-04-01 00:00:00 +0000\n@@ -0,0 +1,1 @@\n+file2 contents at rev 1\n\n")