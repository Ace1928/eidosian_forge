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
def test_legacyScript(self):
    from pyflakes.scripts import pyflakes as script_pyflakes
    self.assertIs(script_pyflakes.checkPath, checkPath)