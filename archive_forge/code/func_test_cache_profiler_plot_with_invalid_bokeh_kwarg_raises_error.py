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
@pytest.mark.skipif('not bokeh')
def test_cache_profiler_plot_with_invalid_bokeh_kwarg_raises_error():
    with CacheProfiler(metric_name='non-standard') as cprof:
        get(dsk, 'e')
    with pytest.raises(AttributeError, match='foo_bar'):
        cprof.visualize(foo_bar='fake')