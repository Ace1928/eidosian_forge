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
def test_exec_nonascii_file(self):
    with tempfile.NamedTemporaryFile(mode='w') as f:
        f.write(dedent('                # coding: utf-8\n                "你好 # nonascii"\n                '))
        f.flush()
        _, stderr = run_with_tty([sys.executable, '-m', 'bpython.curtsies', f.name])
        self.assertEqual(len(stderr), 0)