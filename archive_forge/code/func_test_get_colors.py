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
def test_get_colors():
    from bokeh.palettes import Blues5, Blues256, Viridis
    from dask.diagnostics.profile_visualize import get_colors
    funcs = list(range(11))
    cmap = get_colors('Blues', funcs)
    assert set(cmap) < set(Blues256)
    assert len(set(cmap)) == 11
    funcs = list(range(5))
    cmap = get_colors('Blues', funcs)
    lk = dict(zip(funcs, Blues5))
    assert cmap == [lk[i] for i in funcs]
    funcs = [0, 1, 0, 1, 0, 1]
    cmap = get_colors('BrBG', funcs)
    assert len(set(cmap)) == 2
    funcs = list(range(100))
    cmap = get_colors('Viridis', funcs)
    assert len(set(cmap)) == 100
    funcs = list(range(300))
    cmap = get_colors('Viridis', funcs)
    assert len(set(cmap)) == len(set(Viridis[256]))