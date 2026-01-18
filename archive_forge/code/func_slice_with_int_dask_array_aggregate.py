from __future__ import annotations
import contextlib
from collections.abc import Container, Iterable, Sequence
from functools import wraps
from numbers import Integral
import numpy as np
from tlz import concat
from dask.core import flatten
def slice_with_int_dask_array_aggregate(idx, chunk_outputs, x_chunks, axis):
    """Final aggregation function of `slice_with_int_dask_array_on_axis`.
    Aggregate all chunks of x by one chunk of idx, reordering the output of
    `slice_with_int_dask_array`.

    Note that there is no combine function, as a recursive aggregation (e.g.
    with split_every) would not give any benefit.

    Parameters
    ----------
    idx: ndarray, ndim=1, dtype=any integer
        j-th chunk of idx
    chunk_outputs: ndarray
        concatenation along axis of the outputs of `slice_with_int_dask_array`
        for all chunks of x and the j-th chunk of idx
    x_chunks: tuple
        dask chunks of the x da.Array along axis, e.g. ``(3, 3, 2)``
    axis: int
        normalized axis to take elements from (0 <= axis < x.ndim)

    Returns
    -------
    Selection from all chunks of x for the j-th chunk of idx, in the correct
    order
    """
    idx = idx.astype(np.int64)
    idx = np.where(idx < 0, idx + sum(x_chunks), idx)
    x_chunk_offset = 0
    chunk_output_offset = 0
    idx_final = np.zeros_like(idx)
    for x_chunk in x_chunks:
        idx_filter = (idx >= x_chunk_offset) & (idx < x_chunk_offset + x_chunk)
        idx_cum = np.cumsum(idx_filter)
        idx_final += np.where(idx_filter, idx_cum - 1 + chunk_output_offset, 0)
        x_chunk_offset += x_chunk
        if idx_cum.size > 0:
            chunk_output_offset += idx_cum[-1]
    return chunk_outputs[tuple((idx_final if i == axis else slice(None) for i in range(chunk_outputs.ndim)))]