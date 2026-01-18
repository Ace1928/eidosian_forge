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
def test_checkRecursive(self):
    """
        L{checkRecursive} descends into each directory, finding Python files
        and reporting problems.
        """
    tempdir = tempfile.mkdtemp()
    try:
        os.mkdir(os.path.join(tempdir, 'foo'))
        file1 = os.path.join(tempdir, 'foo', 'bar.py')
        with open(file1, 'wb') as fd:
            fd.write(b'import baz\n')
        file2 = os.path.join(tempdir, 'baz.py')
        with open(file2, 'wb') as fd:
            fd.write(b'import contraband')
        log = []
        reporter = LoggingReporter(log)
        warnings = checkRecursive([tempdir], reporter)
        self.assertEqual(warnings, 2)
        self.assertEqual(sorted(log), sorted([('flake', str(UnusedImport(file1, Node(1), 'baz'))), ('flake', str(UnusedImport(file2, Node(1), 'contraband')))]))
    finally:
        shutil.rmtree(tempdir)