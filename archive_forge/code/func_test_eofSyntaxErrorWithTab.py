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
def test_eofSyntaxErrorWithTab(self):
    """
        The error reported for source files which end prematurely causing a
        syntax error reflects the cause for the syntax error.
        """
    with self.makeTempFile('if True:\n\tfoo =') as sourcePath:
        self.assertHasErrors(sourcePath, [f'{sourcePath}:2:7: invalid syntax\n\tfoo =\n\t     ^\n'])