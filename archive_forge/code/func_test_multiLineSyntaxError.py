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
def test_multiLineSyntaxError(self):
    """
        If there's a multi-line syntax error, then we only report the last
        line.  The offset is adjusted so that it is relative to the start of
        the last line.
        """
    err = io.StringIO()
    lines = ['bad line of source', 'more bad lines of source']
    reporter = Reporter(None, err)
    reporter.syntaxError('foo.py', 'a problem', 3, len(lines[0]) + 7, '\n'.join(lines))
    self.assertEqual('foo.py:3:25: a problem\n' + lines[-1] + '\n' + ' ' * 24 + '^\n', err.getvalue())