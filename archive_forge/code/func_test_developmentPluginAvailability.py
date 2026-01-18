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
def test_developmentPluginAvailability(self) -> None:
    """
        Plugins added in the development path should be loadable, even when
        the (now non-importable) system path contains its own idea of the
        list of plugins for a package.  Inversely, plugins added in the
        system path should not be available.
        """
    for x in range(3):
        names = self.getAllPlugins()
        names.sort()
        self.assertEqual(names, ['app', 'dev'])