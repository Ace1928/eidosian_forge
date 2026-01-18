import errno
import os
import pty
import re
import select
import subprocess
import sys
import tempfile
import unittest
from textwrap import dedent
from bpython import args
from bpython.config import getpreferredencoding
from bpython.test import FixLanguageTestCase as TestCase
def test_exec_nonascii_file_linenums(self):
    with tempfile.NamedTemporaryFile(mode='w') as f:
        f.write(dedent('                1/0\n                '))
        f.flush()
        _, stderr = run_with_tty([sys.executable, '-m', 'bpython.curtsies', f.name])
        self.assertIn('line 1', clean_colors(stderr))