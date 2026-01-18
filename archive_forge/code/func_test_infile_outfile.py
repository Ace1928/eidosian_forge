from __future__ import with_statement
import os
import sys
import textwrap
import unittest
import subprocess
import tempfile
def test_infile_outfile(self):
    infile, infile_name = open_temp_file()
    try:
        infile.write(self.data.encode())
        infile.close()
        outfile, outfile_name = open_temp_file()
        try:
            outfile.close()
            self.assertEqual(self.runTool(args=[infile_name, outfile_name]), [])
            with open(outfile_name, 'rb') as f:
                self.assertEqual(f.read().decode('utf8').splitlines(), self.expect.splitlines())
        finally:
            os.unlink(outfile_name)
    finally:
        os.unlink(infile_name)