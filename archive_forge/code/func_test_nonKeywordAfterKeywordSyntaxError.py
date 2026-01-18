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
def test_nonKeywordAfterKeywordSyntaxError(self):
    """
        Source which has a non-keyword argument after a keyword argument should
        include the line number of the syntax error.  However these exceptions
        do not include an offset.
        """
    source = 'foo(bar=baz, bax)\n'
    with self.makeTempFile(source) as sourcePath:
        if sys.version_info >= (3, 9):
            column = 17
        elif not PYPY:
            column = 14
        else:
            column = 13
        last_line = ' ' * (column - 1) + '^\n'
        columnstr = '%d:' % column
        message = 'positional argument follows keyword argument'
        self.assertHasErrors(sourcePath, ['{}:1:{} {}\nfoo(bar=baz, bax)\n{}'.format(sourcePath, columnstr, message, last_line)])