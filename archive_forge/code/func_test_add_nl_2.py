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
def test_add_nl_2(self):
    """diff generates a valid diff for patches that change last line and
        add a newline.
        """
    lines = udiff_lines([b'boo'], [b'goo\n'])
    self.check_patch(lines)
    self.assertEqual(lines[4], b'\\ No newline at end of file\n')