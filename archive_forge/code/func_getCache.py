import os
import pickle
import sys
import types
from typing import Iterable, Optional, Type, TypeVar
from zope.interface import Interface, providedBy
from twisted.python import log
from twisted.python.components import getAdapterFactory
from twisted.python.modules import getModule
from twisted.python.reflect import namedAny
def getCache(module):
    """
    Compute all the possible loadable plugins, while loading as few as
    possible and hitting the filesystem as little as possible.

    @param module: a Python module object.  This represents a package to search
    for plugins.

    @return: a dictionary mapping module names to L{CachedDropin} instances.
    """
    allCachesCombined = {}
    mod = getModule(module.__name__)
    buckets = {}
    for plugmod in mod.iterModules():
        fpp = plugmod.filePath.parent()
        if fpp not in buckets:
            buckets[fpp] = []
        bucket = buckets[fpp]
        bucket.append(plugmod)
    for pseudoPackagePath, bucket in buckets.items():
        dropinPath = pseudoPackagePath.child('dropin.cache')
        try:
            lastCached = dropinPath.getModificationTime()
            with dropinPath.open('r') as f:
                dropinDotCache = pickle.load(f)
        except BaseException:
            dropinDotCache = {}
            lastCached = 0
        needsWrite = False
        existingKeys = {}
        for pluginModule in bucket:
            pluginKey = pluginModule.name.split('.')[-1]
            existingKeys[pluginKey] = True
            if pluginKey not in dropinDotCache or pluginModule.filePath.getModificationTime() >= lastCached:
                needsWrite = True
                try:
                    provider = pluginModule.load()
                except BaseException:
                    log.err()
                else:
                    entry = _generateCacheEntry(provider)
                    dropinDotCache[pluginKey] = entry
        for pluginKey in list(dropinDotCache.keys()):
            if pluginKey not in existingKeys:
                del dropinDotCache[pluginKey]
                needsWrite = True
        if needsWrite:
            try:
                dropinPath.setContent(pickle.dumps(dropinDotCache))
            except OSError as e:
                log.msg(format='Unable to write to plugin cache %(path)s: error number %(errno)d', path=dropinPath.path, errno=e.errno)
            except BaseException:
                log.err(None, 'Unexpected error while writing cache file')
        allCachesCombined.update(dropinDotCache)
    return allCachesCombined