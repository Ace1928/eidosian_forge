from __future__ import annotations
import contextlib
import os
import warnings
from operator import add, mul
from timeit import default_timer
import pytest
from dask.diagnostics import CacheProfiler, Profiler, ResourceProfiler
from dask.diagnostics.profile_visualize import BOKEH_VERSION
from dask.threaded import get
from dask.utils import apply, tmpfile
from dask.utils_test import slowadd
def test_cache_profiler():
    with CacheProfiler() as cprof:
        in_context_time = default_timer()
        get(dsk2, 'c')
    results = cprof.results
    assert all((isinstance(i, tuple) and len(i) == 5 for i in results))
    assert cprof.start_time < in_context_time < cprof.end_time
    cprof.clear()
    assert cprof.results == []
    tics = [0]

    def nbytes(res):
        tics[0] += 1
        return tics[0]
    with CacheProfiler(nbytes) as cprof:
        get(dsk2, 'c')
    results = cprof.results
    assert tics[-1] == len(results)
    assert tics[-1] == results[-1].metric
    assert cprof._metric_name == 'nbytes'
    assert CacheProfiler(metric=nbytes, metric_name='foo')._metric_name == 'foo'