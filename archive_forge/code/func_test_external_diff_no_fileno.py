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
def test_external_diff_no_fileno(self):
    lines = external_udiff_lines([b'boo\n'] * 10000, [b'goo\n'] * 10000, use_stringio=True)
    self.check_patch(lines)