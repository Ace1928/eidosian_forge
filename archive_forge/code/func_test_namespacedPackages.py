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
def test_namespacedPackages(self) -> None:
    """
        Duplicate packages are not yielded when iterating over namespace
        packages.
        """
    __import__('pkgutil')
    namespaceBoilerplate = b'import pkgutil; __path__ = pkgutil.extend_path(__path__, __name__)'
    entry = self.pathEntryWithOnePackage()
    testPackagePath = entry.child('test_package')
    testPackagePath.child('__init__.py').setContent(namespaceBoilerplate)
    nestedEntry = testPackagePath.child('nested_package')
    nestedEntry.makedirs()
    nestedEntry.child('__init__.py').setContent(namespaceBoilerplate)
    nestedEntry.child('module.py').setContent(b'')
    anotherEntry = self.pathEntryWithOnePackage()
    anotherPackagePath = anotherEntry.child('test_package')
    anotherPackagePath.child('__init__.py').setContent(namespaceBoilerplate)
    anotherNestedEntry = anotherPackagePath.child('nested_package')
    anotherNestedEntry.makedirs()
    anotherNestedEntry.child('__init__.py').setContent(namespaceBoilerplate)
    anotherNestedEntry.child('module2.py').setContent(b'')
    self.replaceSysPath([entry.path, anotherEntry.path])
    module = modules.getModule('test_package')
    try:
        walkedNames = [mod.name for mod in module.walkModules(importPackages=True)]
    finally:
        for module in list(sys.modules.keys()):
            if module.startswith('test_package'):
                del sys.modules[module]
    expected = ['test_package', 'test_package.nested_package', 'test_package.nested_package.module', 'test_package.nested_package.module2']
    self.assertEqual(walkedNames, expected)