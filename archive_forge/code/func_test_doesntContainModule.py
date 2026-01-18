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
def test_doesntContainModule(self) -> None:
    """
        L{PythonPath} implements the C{in} operator so that when it is the
        right-hand argument and the name of a module which does not exist on
        that L{PythonPath} is the left-hand argument, the result is C{False}.
        """
    thePath = modules.PythonPath()
    self.assertNotIn('bogusModule', thePath)