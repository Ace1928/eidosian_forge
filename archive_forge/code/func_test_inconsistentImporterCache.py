from __future__ import annotations
import compileall
import itertools
import sys
import zipfile
from importlib.abc import PathEntryFinder
from types import ModuleType
from typing import Any, Generator
from typing_extensions import Protocol
import twisted
from twisted.python import modules
from twisted.python.compat import networkString
from twisted.python.filepath import FilePath
from twisted.python.reflect import namedAny
from twisted.python.test.modules_helpers import TwistedModulesMixin
from twisted.python.test.test_zippath import zipit
from twisted.trial.unittest import TestCase
def test_inconsistentImporterCache(self) -> None:
    """
        If the path a module loaded with L{PythonPath.__getitem__} is not
        present in the path importer cache, a warning is emitted, but the
        L{PythonModule} is returned as usual.
        """
    space = modules.PythonPath([], sys.modules, [], {})
    thisModule = space[__name__]
    warnings = self.flushWarnings([self.test_inconsistentImporterCache])
    self.assertEqual(warnings[0]['category'], UserWarning)
    self.assertEqual(warnings[0]['message'], FilePath(twisted.__file__).parent().dirname() + ' (for module ' + __name__ + ') not in path importer cache (PEP 302 violation - check your local configuration).')
    self.assertEqual(len(warnings), 1)
    self.assertEqual(thisModule.name, __name__)