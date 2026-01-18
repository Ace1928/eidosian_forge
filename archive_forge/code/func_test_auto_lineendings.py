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
def test_auto_lineendings(self):
    """
    Verify that windows line-endings are detected and preserved on input.
    """
    thisdir = os.path.realpath(os.path.dirname(__file__))
    for suffix in ['win', 'unix']:
        with self.subTest(lineending=suffix):
            filename = 'test_lineend_{}.cmake'.format(suffix)
            infile_path = os.path.join(thisdir, 'testdata', filename)
            outfile_path = os.path.join(self.tempdir, filename)
            subprocess.check_call([sys.executable, '-Bm', 'cmakelang.format', '--line-ending=auto', '-o', outfile_path, infile_path], cwd=self.tempdir, env=self.env)
            with open(infile_path, 'rb') as infile:
                infile_bytes = infile.read()
            with open(outfile_path, 'rb') as infile:
                outfile_bytes = infile.read()
            self.assertEqual(infile_bytes, outfile_bytes)