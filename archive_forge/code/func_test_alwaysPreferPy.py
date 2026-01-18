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
def test_alwaysPreferPy(self) -> None:
    """
        Verify that .py files will always be preferred to .pyc files, regardless of
        directory listing order.
        """
    mypath = FilePath(self.mktemp())
    mypath.createDirectory()
    pp = modules.PythonPath(sysPath=[mypath.path])
    originalSmartPath = pp._smartPath

    def _evilSmartPath(pathName: str) -> Any:
        o = originalSmartPath(pathName)
        originalChildren = o.children

        def evilChildren() -> Any:
            x = list(originalChildren())
            x.sort()
            x.reverse()
            return x
        o.children = evilChildren
        return o
    mypath.child('abcd.py').setContent(b'\n')
    compileall.compile_dir(mypath.path, quiet=True)
    self.assertEqual(len(list(mypath.children())), 2)
    pp._smartPath = _evilSmartPath
    self.assertEqual(pp['abcd'].filePath, mypath.child('abcd.py'))