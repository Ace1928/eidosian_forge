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
def test_errors_syntax(self):
    """
        When pyflakes finds errors with the files it's given, (if they don't
        exist, say), then the return code is non-zero and the errors are
        printed to stderr.
        """
    with open(self.tempfilepath, 'wb') as fd:
        fd.write(b'import')
    d = self.runPyflakes([self.tempfilepath])
    error_msg = '{0}:1:7: invalid syntax{1}import{1}      ^{1}'.format(self.tempfilepath, os.linesep)
    self.assertEqual(d, ('', error_msg, 1))