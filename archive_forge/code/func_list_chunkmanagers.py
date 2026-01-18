from __future__ import annotations
import functools
import sys
from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from importlib.metadata import EntryPoint, entry_points
from typing import TYPE_CHECKING, Any, Callable, Generic, Protocol, TypeVar
import numpy as np
from xarray.core.utils import emit_user_level_warning
from xarray.namedarray.pycompat import is_chunked_array
@functools.lru_cache(maxsize=1)
def list_chunkmanagers() -> dict[str, ChunkManagerEntrypoint[Any]]:
    """
    Return a dictionary of available chunk managers and their ChunkManagerEntrypoint subclass objects.

    Returns
    -------
    chunkmanagers : dict
        Dictionary whose values are registered ChunkManagerEntrypoint subclass instances, and whose values
        are the strings under which they are registered.

    Notes
    -----
    # New selection mechanism introduced with Python 3.10. See GH6514.
    """
    if sys.version_info >= (3, 10):
        entrypoints = entry_points(group='xarray.chunkmanagers')
    else:
        entrypoints = entry_points().get('xarray.chunkmanagers', ())
    return load_chunkmanagers(entrypoints)