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
def test_multilineSyntaxError(self):
    """
        Source which includes a syntax error which results in the raised
        L{SyntaxError.text} containing multiple lines of source are reported
        with only the last line of that source.
        """
    source = "def foo():\n    '''\n\ndef bar():\n    pass\n\ndef baz():\n    '''quux'''\n"

    def evaluate(source):
        exec(source)
    try:
        evaluate(source)
    except SyntaxError as e:
        if not PYPY and sys.version_info < (3, 10):
            self.assertTrue(e.text.count('\n') > 1)
    else:
        self.fail()
    with self.makeTempFile(source) as sourcePath:
        if PYPY:
            message = 'end of file (EOF) while scanning triple-quoted string literal'
        elif sys.version_info >= (3, 10):
            message = 'unterminated triple-quoted string literal (detected at line 8)'
        else:
            message = 'invalid syntax'
        if PYPY or sys.version_info >= (3, 10):
            column = 12
        else:
            column = 8
        self.assertHasErrors(sourcePath, ["%s:8:%d: %s\n    '''quux'''\n%s^\n" % (sourcePath, column, message, ' ' * (column - 1))])