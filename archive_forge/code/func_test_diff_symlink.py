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
def test_diff_symlink(self):
    differ = diff.DiffSymlink(self.old_tree, self.new_tree, BytesIO())
    differ.diff_symlink('old target', None)
    self.assertEqual(b"=== target was 'old target'\n", differ.to_file.getvalue())
    differ = diff.DiffSymlink(self.old_tree, self.new_tree, BytesIO())
    differ.diff_symlink(None, 'new target')
    self.assertEqual(b"=== target is 'new target'\n", differ.to_file.getvalue())
    differ = diff.DiffSymlink(self.old_tree, self.new_tree, BytesIO())
    differ.diff_symlink('old target', 'new target')
    self.assertEqual(b"=== target changed 'old target' => 'new target'\n", differ.to_file.getvalue())