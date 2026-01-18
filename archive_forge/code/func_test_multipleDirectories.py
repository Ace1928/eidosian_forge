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
def test_multipleDirectories(self):
    """
        L{iterSourceCode} can be given multiple directories.  It will recurse
        into each of them.
        """
    foopath = os.path.join(self.tempdir, 'foo')
    barpath = os.path.join(self.tempdir, 'bar')
    os.mkdir(foopath)
    apath = self.makeEmptyFile('foo', 'a.py')
    os.mkdir(barpath)
    bpath = self.makeEmptyFile('bar', 'b.py')
    self.assertEqual(sorted(iterSourceCode([foopath, barpath])), sorted([apath, bpath]))