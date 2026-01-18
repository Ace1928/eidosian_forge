from datashader.compiler import compile_components
from datashader.utils import Dispatcher
from datashader.glyphs.line import LinesXarrayCommonX
from datashader.glyphs.quadmesh import (
from datashader.utils import apply
import dask
import numpy as np
import xarray as xr
from dask.base import tokenize, compute
from dask.array.overlap import overlap
def dask_xarray_pipeline(glyph, xr_ds, schema, canvas, summary, *, antialias=False, cuda=False):
    dsk, name = dask_glyph_dispatch(glyph, xr_ds, schema, canvas, summary, antialias=antialias, cuda=cuda)
    scheduler = dask.base.get_scheduler() or xr_ds.__dask_scheduler__
    keys = xr_ds.__dask_keys__()
    optimize = xr_ds.__dask_optimize__
    graph = xr_ds.__dask_graph__()
    dsk.update(optimize(graph, keys))
    return scheduler(dsk, name)