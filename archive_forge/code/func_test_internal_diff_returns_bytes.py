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
def test_internal_diff_returns_bytes(self):
    output = StubO()
    diff.internal_diff('old_µ', [b'old_text\n'], 'new_å', [b'new_text\n'], output)
    output.check_types(self, bytes)