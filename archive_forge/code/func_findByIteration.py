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
def findByIteration(self, modname: str, where: _SupportsWalkModules=modules, importPackages: bool=False) -> modules.PythonModule:
    """
        You don't ever actually want to do this, so it's not in the public
        API, but sometimes we want to compare the result of an iterative call
        with a lookup call and make sure they're the same for test purposes.
        """
    for modinfo in where.walkModules(importPackages=importPackages):
        if modinfo.name == modname:
            return modinfo
    self.fail(f'Unable to find module {modname!r} through iteration.')