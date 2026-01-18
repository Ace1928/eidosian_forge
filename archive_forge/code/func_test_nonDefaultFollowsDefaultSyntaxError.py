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
def test_nonDefaultFollowsDefaultSyntaxError(self):
    """
        Source which has a non-default argument following a default argument
        should include the line number of the syntax error.  However these
        exceptions do not include an offset.
        """
    source = 'def foo(bar=baz, bax):\n    pass\n'
    with self.makeTempFile(source) as sourcePath:
        if sys.version_info >= (3, 12):
            msg = 'parameter without a default follows parameter with a default'
        else:
            msg = 'non-default argument follows default argument'
        if PYPY and sys.version_info >= (3, 9):
            column = 18
        elif PYPY:
            column = 8
        elif sys.version_info >= (3, 10):
            column = 18
        elif sys.version_info >= (3, 9):
            column = 21
        else:
            column = 9
        last_line = ' ' * (column - 1) + '^\n'
        self.assertHasErrors(sourcePath, [f'{sourcePath}:1:{column}: {msg}\ndef foo(bar=baz, bax):\n{last_line}'])