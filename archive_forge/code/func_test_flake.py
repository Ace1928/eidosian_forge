import contextlib
import io
import os
import sys
import shutil
import subprocess
import tempfile
from pyflakes.checker import PYPY
from pyflakes.messages import UnusedImport
from pyflakes.reporter import Reporter
from pyflakes.api import (
from pyflakes.test.harness import TestCase, skipIf
def test_flake(self):
    """
        C{flake} reports a code warning from Pyflakes.  It is exactly the
        str() of a L{pyflakes.messages.Message}.
        """
    out = io.StringIO()
    reporter = Reporter(out, None)
    message = UnusedImport('foo.py', Node(42), 'bar')
    reporter.flake(message)
    self.assertEqual(out.getvalue(), f'{message}\n')