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
def test_encoding_invocation(self):
    """
    Try to reformat latin1-encoded file, once with default
    encoding (-> prompt utf8-decoding error) and once with
    specifically latin1 encoding (-> should succeed)
    """
    thisdir = os.path.realpath(os.path.dirname(__file__))
    infile_path = os.path.join(thisdir, 'testdata', 'test_latin1_in.cmake')
    expectfile_path = os.path.join(thisdir, 'testdata', 'test_latin1_out.cmake')
    invocation_result = subprocess.call([sys.executable, '-Bm', 'cmakelang.format', '--outfile-path', os.path.join(self.tempdir, 'test_latin1_out.cmake'), infile_path], cwd=self.tempdir, env=self.env, stderr=subprocess.PIPE)
    self.assertNotEqual(0, invocation_result, msg='Expected cmake-format invocation to fail but did not')
    subprocess.check_call([sys.executable, '-Bm', 'cmakelang.format', '--input-encoding=latin1', '--output-encoding=latin1', '--outfile-path', os.path.join(self.tempdir, 'test_latin1_out.cmake'), infile_path], cwd=self.tempdir, env=self.env)
    with io.open(os.path.join(self.tempdir, 'test_latin1_out.cmake'), 'r', encoding='latin1') as infile:
        actual_text = infile.read()
    with io.open(expectfile_path, 'r', encoding='latin1') as infile:
        expected_text = infile.read()
    delta_lines = list(difflib.unified_diff(expected_text.split('\n'), actual_text.split('\n')))
    if delta_lines:
        raise AssertionError('\n'.join(delta_lines[2:]))