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
def test_misencodedFileUTF16(self):
    """
        If a source file contains bytes which cannot be decoded, this is
        reported on stderr.
        """
    SNOWMAN = chr(9731)
    source = ('# coding: ascii\nx = "%s"\n' % SNOWMAN).encode('utf-16')
    with self.makeTempFile(source) as sourcePath:
        if sys.version_info < (3, 11, 4):
            expected = f'{sourcePath}: problem decoding source\n'
        else:
            expected = f'{sourcePath}:1: source code string cannot contain null bytes\n'
        self.assertHasErrors(sourcePath, [expected])