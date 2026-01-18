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
def dask_rectilinear(glyph, xr_ds, schema, canvas, summary, *, antialias=False, cuda=False):
    shape, bounds, st, axis = shape_bounds_st_and_axis(xr_ds, canvas, glyph)
    create, info, append, combine, finalize, antialias_stage_2, antialias_stage_2_funcs, _ = compile_components(summary, schema, glyph, antialias=antialias, cuda=cuda, partitioned=True)
    x_mapper = canvas.x_axis.mapper
    y_mapper = canvas.y_axis.mapper
    extend = glyph._build_extend(x_mapper, y_mapper, info, append, antialias_stage_2, antialias_stage_2_funcs)
    x_range = bounds[:2]
    y_range = bounds[2:]
    chunk_inds = {}
    for k, chunks in xr_ds.chunks.items():
        chunk_inds[k] = [0] + list(np.cumsum(chunks))
    x_name = glyph.x
    y_name = glyph.y
    coords = xr_ds[glyph.name].coords
    coord_dims = list(coords.dims)
    xdim_ind = coord_dims.index(x_name)
    ydim_ind = coord_dims.index(y_name)
    var_name = list(xr_ds.data_vars.keys())[0]
    xs = xr_ds[x_name].values
    ys = xr_ds[y_name].values
    x_breaks = glyph.infer_interval_breaks(xs)
    y_breaks = glyph.infer_interval_breaks(ys)

    def chunk(np_arr, *inds):
        chunk_coords_list = []
        for i, coord_name in enumerate(coords.dims):
            chunk_number = inds[i]
            coord_slice = slice(chunk_inds[coord_name][chunk_number], chunk_inds[coord_name][chunk_number + 1])
            chunk_coords_list.append([coord_name, coords[coord_name][coord_slice]])
        chunk_coords = dict(chunk_coords_list)
        chunk_ds = xr.DataArray(np_arr, coords=chunk_coords, dims=coord_dims, name=var_name).to_dataset()
        x_chunk_number = inds[xdim_ind]
        x_breaks_slice = slice(chunk_inds[x_name][x_chunk_number], chunk_inds[x_name][x_chunk_number + 1] + 1)
        x_breaks_chunk = x_breaks[x_breaks_slice]
        y_chunk_number = inds[ydim_ind]
        y_breaks_slice = slice(chunk_inds[y_name][y_chunk_number], chunk_inds[y_name][y_chunk_number + 1] + 1)
        y_breaks_chunk = y_breaks[y_breaks_slice]
        aggs = create(shape)
        extend(aggs, chunk_ds, st, bounds, x_breaks=x_breaks_chunk, y_breaks=y_breaks_chunk)
        return aggs
    name = tokenize(xr_ds.__dask_tokenize__(), canvas, glyph, summary)
    keys = [k for row in xr_ds.__dask_keys__()[0] for k in row]
    keys2 = [(name, i) for i in range(len(keys))]
    dsk = dict(((k2, (chunk, k, k[1], k[2])) for k2, k in zip(keys2, keys)))
    dsk[name] = (apply, finalize, [(combine, keys2)], dict(cuda=cuda, coords=axis, dims=[glyph.y_label, glyph.x_label], attrs=dict(x_range=x_range, y_range=y_range)))
    return (dsk, name)