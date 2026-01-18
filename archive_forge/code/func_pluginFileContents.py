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
def pluginFileContents(name: str) -> bytes:
    return 'from zope.interface import provider\nfrom twisted.plugin import IPlugin\nfrom twisted.test.test_plugin import ITestPlugin\n\n@provider(IPlugin, ITestPlugin)\nclass {}:\n    pass\n'.format(name).encode('ascii')