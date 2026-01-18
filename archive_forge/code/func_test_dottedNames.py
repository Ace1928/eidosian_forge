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
def test_dottedNames(self) -> None:
    """
        Verify that the walkModules APIs will give us back subpackages, not just
        subpackages.
        """
    self.assertEqual(modules.getModule('twisted.python'), self.findByIteration('twisted.python', where=modules.getModule('twisted')))