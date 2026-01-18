from __future__ import annotations
import re
import sys
import warnings
from functools import wraps, lru_cache
from itertools import count
from typing import TYPE_CHECKING, Generic, Iterator, NamedTuple, TypeVar, TypedDict, overload
@lru_cache(maxsize=None)
def get_installed_extensions():
    """ Return all entry_points in the `markdown.extensions` group. """
    if sys.version_info >= (3, 10):
        from importlib import metadata
    else:
        import importlib_metadata as metadata
    return metadata.entry_points(group='markdown.extensions')