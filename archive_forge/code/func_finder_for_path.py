from __future__ import unicode_literals
import bisect
import io
import logging
import os
import pkgutil
import sys
import types
import zipimport
from . import DistlibException
from .util import cached_property, get_cache_base, Cache
def finder_for_path(path):
    """
    Return a resource finder for a path, which should represent a container.

    :param path: The path.
    :return: A :class:`ResourceFinder` instance for the path.
    """
    result = None
    pkgutil.get_importer(path)
    loader = sys.path_importer_cache.get(path)
    finder = _finder_registry.get(type(loader))
    if finder:
        module = _dummy_module
        module.__file__ = os.path.join(path, '')
        module.__loader__ = loader
        result = finder(module)
    return result