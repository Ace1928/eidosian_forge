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
def test_deployedMode(self) -> None:
    """
        The C{dropin.cache} file may not be writable: the cache should still be
        attainable, but an error should be logged to show that the cache
        couldn't be updated.
        """
    plugin.getCache(self.module)
    cachepath = self.package.child('dropin.cache')
    FilePath(__file__).sibling('plugin_extra1.py').copyTo(self.package.child('pluginextra.py'))
    invalidateImportCaches()
    os.chmod(self.package.path, 320)
    os.chmod(cachepath.path, 256)
    self.addCleanup(os.chmod, self.package.path, 448)
    self.addCleanup(os.chmod, cachepath.path, 448)
    events: list[EventDict] = []
    addObserver(events.append)
    self.addCleanup(removeObserver, events.append)
    cache = plugin.getCache(self.module)
    self.assertIn('pluginextra', cache)
    self.assertIn(self.originalPlugin, cache)
    expected = 'Unable to write to plugin cache %s: error number %d' % (cachepath.path, errno.EPERM)
    for event in events:
        maybeText = textFromEventDict(event)
        assert maybeText is not None
        if expected in maybeText:
            break
    else:
        self.fail('Did not observe unwriteable cache warning in log events: %r' % (events,))