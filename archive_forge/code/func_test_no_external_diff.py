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
def test_no_external_diff(self):
    """Check that NoDiff is raised when diff is not available"""
    self.overrideEnv('PATH', '')
    self.assertRaises(errors.NoDiff, diff.external_diff, b'old', [b'boo\n'], b'new', [b'goo\n'], BytesIO(), diff_opts=['-u'])