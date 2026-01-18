from __future__ import annotations
from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, Any, Callable
import numpy as np
from packaging.version import Version
from xarray.core.indexing import ImplicitToExplicitIndexingAdapter
from xarray.namedarray.parallelcompat import ChunkManagerEntrypoint, T_ChunkedArray
from xarray.namedarray.utils import is_duck_dask_array, module_available
def blockwise(self, func: Callable[..., Any], out_ind: Iterable[Any], *args: Any, name: str | None=None, token: Any | None=None, dtype: _DType_co | None=None, adjust_chunks: dict[Any, Callable[..., Any]] | None=None, new_axes: dict[Any, int] | None=None, align_arrays: bool=True, concatenate: bool | None=None, meta: tuple[np.ndarray[Any, _DType_co], ...] | None=None, **kwargs: Any) -> DaskArray | Any:
    from dask.array import blockwise
    return blockwise(func, out_ind, *args, name=name, token=token, dtype=dtype, adjust_chunks=adjust_chunks, new_axes=new_axes, align_arrays=align_arrays, concatenate=concatenate, meta=meta, **kwargs)