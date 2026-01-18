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
def test_internal_diff_iso_8859_1(self):
    output = BytesIO()
    diff.internal_diff('old_µ', [b'old_text\n'], 'new_å', [b'new_text\n'], output, path_encoding='iso-8859-1')
    lines = output.getvalue().splitlines(True)
    self.check_patch(lines)
    self.assertEqual([b'--- old_\xb5\n', b'+++ new_\xe5\n', b'@@ -1,1 +1,1 @@\n', b'-old_text\n', b'+new_text\n', b'\n'], lines)