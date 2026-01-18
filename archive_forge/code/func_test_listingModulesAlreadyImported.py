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
def test_listingModulesAlreadyImported(self) -> None:
    """
        Make sure the module list comes back as we expect from iterModules on a
        package, whether zipped or not, even if the package has already been
        imported.
        """
    self._setupSysPath()
    namedAny(self.packageName)
    self._listModules()