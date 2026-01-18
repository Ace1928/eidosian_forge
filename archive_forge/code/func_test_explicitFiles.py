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
def test_explicitFiles(self):
    """
        If one of the paths given to L{iterSourceCode} is not a directory but
        a file, it will include that in its output.
        """
    epath = self.makeEmptyFile('e.py')
    self.assertEqual(list(iterSourceCode([epath])), [epath])