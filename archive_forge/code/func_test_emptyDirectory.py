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
def test_emptyDirectory(self):
    """
        There are no Python files in an empty directory.
        """
    self.assertEqual(list(iterSourceCode([self.tempdir])), [])