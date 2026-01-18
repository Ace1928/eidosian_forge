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
def test_execute_missing(self):
    diff_obj = diff.DiffFromTool(['a-tool-which-is-unlikely-to-exist'], None, None, None)
    self.addCleanup(diff_obj.finish)
    e = self.assertRaises(errors.ExecutableMissing, diff_obj._execute, 'old', 'new')
    self.assertEqual('a-tool-which-is-unlikely-to-exist could not be found on this machine', str(e))