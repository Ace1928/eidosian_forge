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
def makeEmptyFile(self, *parts):
    assert parts
    fpath = os.path.join(self.tempdir, *parts)
    open(fpath, 'a').close()
    return fpath