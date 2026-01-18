from __future__ import annotations
import sys
import warnings
from io import StringIO
from typing import Mapping, Sequence, TypeVar
from unittest import TestResult
from twisted.python.filepath import FilePath
from twisted.trial._synctest import (
from twisted.trial.unittest import SynchronousTestCase
import warnings
import warnings
def test_missingSource(self) -> None:
    """
        Warnings emitted by a function the source code of which is not
        available can still be flushed.
        """
    package = FilePath(self.mktemp().encode('utf-8')).child(b'twisted_private_helper')
    package.makedirs()
    package.child(b'__init__.py').setContent(b'')
    package.child(b'missingsourcefile.py').setContent(b'\nimport warnings\ndef foo():\n    warnings.warn("oh no")\n')
    pathEntry = package.parent().path.decode('utf-8')
    sys.path.insert(0, pathEntry)
    self.addCleanup(sys.path.remove, pathEntry)
    from twisted_private_helper import missingsourcefile
    self.addCleanup(sys.modules.pop, 'twisted_private_helper')
    self.addCleanup(sys.modules.pop, missingsourcefile.__name__)
    package.child(b'missingsourcefile.py').remove()
    missingsourcefile.foo()
    self.assertEqual(len(self.flushWarnings([missingsourcefile.foo])), 1)