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
def test_diff_deletion(self):
    self.build_tree_contents([('old-tree/file', b'contents'), ('new-tree/file', b'contents')])
    self.old_tree.add('file', ids=b'file-id')
    self.new_tree.add('file', ids=b'file-id')
    os.unlink('new-tree/file')
    self.differ.show_diff(None)
    self.assertContainsRe(self.differ.to_file.getvalue(), b'-contents')