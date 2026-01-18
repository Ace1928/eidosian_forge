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
@_withCacheness
def test_detectFilesChanged(self) -> None:
    """
        Check that if the content of a plugin change, L{plugin.getPlugins} is
        able to detect the new plugins added.
        """
    FilePath(__file__).sibling('plugin_extra1.py').copyTo(self.package.child('pluginextra.py'))
    try:
        plgs = list(plugin.getPlugins(ITestPlugin, self.module))
        self.assertEqual(len(plgs), 2)
        FilePath(__file__).sibling('plugin_extra2.py').copyTo(self.package.child('pluginextra.py'))
        self._unimportPythonModule(sys.modules['mypackage.pluginextra'])
        plgs = list(plugin.getPlugins(ITestPlugin, self.module))
        self.assertEqual(len(plgs), 3)
        names = ['TestPlugin', 'FourthTestPlugin', 'FifthTestPlugin']
        for p in plgs:
            names.remove(p.__name__)
            p.test1()
    finally:
        self._unimportPythonModule(sys.modules['mypackage.pluginextra'], True)