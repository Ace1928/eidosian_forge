from __future__ import annotations
import compileall
import errno
import functools
import os
import sys
import time
from importlib import invalidate_caches as invalidateImportCaches
from types import ModuleType
from typing import Callable, TypedDict, TypeVar
from zope.interface import Interface
from twisted import plugin
from twisted.python.filepath import FilePath
from twisted.python.log import EventDict, addObserver, removeObserver, textFromEventDict
from twisted.trial import unittest
from twisted.plugin import pluginPackagePaths
def createDummyPackage(self, root: FilePath[str], name: str, pluginName: str) -> FilePath[str]:
    """
        Create a directory containing a Python package named I{dummy} with a
        I{plugins} subpackage.

        @type root: L{FilePath}
        @param root: The directory in which to create the hierarchy.

        @type name: C{str}
        @param name: The name of the directory to create which will contain
            the package.

        @type pluginName: C{str}
        @param pluginName: The name of a module to create in the
            I{dummy.plugins} package.

        @rtype: L{FilePath}
        @return: The directory which was created to contain the I{dummy}
            package.
        """
    directory = root.child(name)
    package = directory.child('dummy')
    package.makedirs()
    package.child('__init__.py').setContent(b'')
    plugins = package.child('plugins')
    plugins.makedirs()
    plugins.child('__init__.py').setContent(pluginInitFile)
    pluginModule = plugins.child(pluginName + '.py')
    pluginModule.setContent(pluginFileContents(name))
    return directory