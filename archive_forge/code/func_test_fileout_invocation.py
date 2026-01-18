from __future__ import unicode_literals
import contextlib
import difflib
import io
import os
import shutil
import subprocess
import sys
import unittest
import tempfile
def test_fileout_invocation(self):
    """
    Test invocation with an infile path and outfile path
    """
    thisdir = os.path.realpath(os.path.dirname(__file__))
    infile_path = os.path.join(thisdir, 'testdata', 'test_in.cmake')
    expectfile_path = os.path.join(thisdir, 'testdata', 'test_out.cmake')
    subprocess.check_call([sys.executable, '-Bm', 'cmakelang.format', '-o', os.path.join(self.tempdir, 'test_out.cmake'), infile_path], cwd=self.tempdir, env=self.env)
    with io.open(os.path.join(self.tempdir, 'test_out.cmake'), 'r', encoding='utf8') as infile:
        actual_text = infile.read()
    with io.open(expectfile_path, 'r', encoding='utf8') as infile:
        expected_text = infile.read()
    delta_lines = list(difflib.unified_diff(expected_text.split('\n'), actual_text.split('\n')))
    if delta_lines:
        raise AssertionError('\n'.join(delta_lines[2:]))