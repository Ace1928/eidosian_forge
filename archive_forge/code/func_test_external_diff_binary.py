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
def test_external_diff_binary(self):
    """The output when using external diff should use diff's i18n error"""
    for lang in ('LANG', 'LC_ALL', 'LANGUAGE'):
        self.overrideEnv(lang, 'C')
    lines = external_udiff_lines([b'\x00foobar\n'], [b'foo\x00bar\n'])
    cmd = ['diff', '-u', '--binary', 'old', 'new']
    with open('old', 'wb') as f:
        f.write(b'\x00foobar\n')
    with open('new', 'wb') as f:
        f.write(b'foo\x00bar\n')
    pipe = subprocess.Popen(cmd, stdout=subprocess.PIPE, stdin=subprocess.PIPE)
    out, err = pipe.communicate()
    self.assertEqual(out.splitlines(True) + [b'\n'], lines)