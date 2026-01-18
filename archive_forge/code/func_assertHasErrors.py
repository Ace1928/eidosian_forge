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
def assertHasErrors(self, path, errorList):
    """
        Assert that C{path} causes errors.

        @param path: A path to a file to check.
        @param errorList: A list of errors expected to be printed to stderr.
        """
    err = io.StringIO()
    count = withStderrTo(err, checkPath, path)
    self.assertEqual((count, err.getvalue()), (len(errorList), ''.join(errorList)))