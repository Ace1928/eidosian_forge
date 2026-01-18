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
def test_execute(self):
    output = BytesIO()
    diff_obj = diff.DiffFromTool([sys.executable, '-c', 'print("{old_path} {new_path}")'], None, None, output)
    self.addCleanup(diff_obj.finish)
    diff_obj._execute('old', 'new')
    self.assertEqual(output.getvalue().rstrip(), b'old new')