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
def test_encodedFileUTF8(self):
    """
        If source file declares the correct encoding, no error is reported.
        """
    SNOWMAN = chr(9731)
    source = ('# coding: utf-8\nx = "%s"\n' % SNOWMAN).encode('utf-8')
    with self.makeTempFile(source) as sourcePath:
        self.assertHasErrors(sourcePath, [])