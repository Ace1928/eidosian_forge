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
def get_chunked_array_type(*args: Any) -> ChunkManagerEntrypoint[Any]:
    """
    Detects which parallel backend should be used for given set of arrays.

    Also checks that all arrays are of same chunking type (i.e. not a mix of cubed and dask).
    """
    ALLOWED_NON_CHUNKED_TYPES = {int, float, np.ndarray}
    chunked_arrays = [a for a in args if is_chunked_array(a) and type(a) not in ALLOWED_NON_CHUNKED_TYPES]
    chunked_array_types = {type(a) for a in chunked_arrays}
    if len(chunked_array_types) > 1:
        raise TypeError(f'Mixing chunked array types is not supported, but received multiple types: {chunked_array_types}')
    elif len(chunked_array_types) == 0:
        raise TypeError('Expected a chunked array but none were found')
    chunked_arr = chunked_arrays[0]
    chunkmanagers = list_chunkmanagers()
    selected = [chunkmanager for chunkmanager in chunkmanagers.values() if chunkmanager.is_chunked_array(chunked_arr)]
    if not selected:
        raise TypeError(f'Could not find a Chunk Manager which recognises type {type(chunked_arr)}')
    elif len(selected) >= 2:
        raise TypeError(f'Multiple ChunkManagers recognise type {type(chunked_arr)}')
    else:
        return selected[0]