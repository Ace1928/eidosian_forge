from __future__ import annotations
from numbers import Number
from math import log10
import warnings
import numpy as np
import pandas as pd
import dask.dataframe as dd
import dask.array as da
from xarray import DataArray, Dataset
from .utils import Dispatcher, ngjit, calc_res, calc_bbox, orient_array, \
from .utils import get_indices, dshape_from_pandas, dshape_from_dask
from .utils import Expr # noqa (API import)
from .resampling import resample_2d, resample_2d_distributed
from . import reductions as rd
def bypixel(source, canvas, glyph, agg, *, antialias=False):
    """Compute an aggregate grouped by pixel sized bins.

    Aggregate input data ``source`` into a grid with shape and axis matching
    ``canvas``, mapping data to bins by ``glyph``, and aggregating by reduction
    ``agg``.

    Parameters
    ----------
    source : pandas.DataFrame, dask.DataFrame
        Input datasource
    canvas : Canvas
    glyph : Glyph
    agg : Reduction
    """
    source, dshape = _bypixel_sanitise(source, glyph, agg)
    schema = dshape.measure
    glyph.validate(schema)
    agg.validate(schema)
    canvas.validate()
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', 'All-NaN (slice|axis) encountered')
        return bypixel.pipeline(source, schema, canvas, glyph, agg, antialias=antialias)