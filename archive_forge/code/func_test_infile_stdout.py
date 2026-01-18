from __future__ import with_statement
import os
import sys
import textwrap
import unittest
import subprocess
import tempfile
def test_infile_stdout(self):
    infile, infile_name = open_temp_file()
    try:
        infile.write(self.data.encode())
        infile.close()
        self.assertEqual(self.runTool(args=[infile_name]), self.expect.splitlines())
    finally:
        os.unlink(infile_name)