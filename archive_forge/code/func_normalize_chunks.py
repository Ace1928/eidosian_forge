from __future__ import annotations
from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, Any, Callable
import numpy as np
from packaging.version import Version
from xarray.core.indexing import ImplicitToExplicitIndexingAdapter
from xarray.namedarray.parallelcompat import ChunkManagerEntrypoint, T_ChunkedArray
from xarray.namedarray.utils import is_duck_dask_array, module_available
def normalize_chunks(self, chunks: T_Chunks | _NormalizedChunks, shape: tuple[int, ...] | None=None, limit: int | None=None, dtype: _DType_co | None=None, previous_chunks: _NormalizedChunks | None=None) -> Any:
    """Called by open_dataset"""
    from dask.array.core import normalize_chunks
    return normalize_chunks(chunks, shape=shape, limit=limit, dtype=dtype, previous_chunks=previous_chunks)