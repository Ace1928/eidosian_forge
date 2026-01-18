from __future__ import annotations
import os
import re
from inspect import getmro
import numba as nb
import numpy as np
import pandas as pd
from toolz import memoize
from xarray import DataArray
import dask.dataframe as dd
import datashader.datashape as datashape
def dshape_from_pandas_helper(col):
    """Return an object from datashader.datashape.coretypes given a column from a pandas
    dataframe.
    """
    if isinstance(col.dtype, type(pd.Categorical.dtype)) or isinstance(col.dtype, pd.api.types.CategoricalDtype) or (cudf and isinstance(col.dtype, cudf.core.dtypes.CategoricalDtype)):
        pd_categories = col.cat.categories
        if isinstance(pd_categories, dd.Index):
            pd_categories = pd_categories.compute()
        if cudf and isinstance(pd_categories, cudf.Index):
            pd_categories = pd_categories.to_pandas()
        categories = np.array(pd_categories)
        if categories.dtype.kind == 'U':
            categories = categories.astype('object')
        cat_dshape = datashape.dshape('{} * {}'.format(len(col.cat.categories), categories.dtype))
        return datashape.Categorical(categories, type=cat_dshape, ordered=col.cat.ordered)
    elif col.dtype.kind == 'M':
        tz = getattr(col.dtype, 'tz', None)
        if tz is not None:
            tz = str(tz)
        return datashape.Option(datashape.DateTime(tz=tz))
    elif isinstance(col.dtype, (RaggedDtype, GeometryDtype)):
        return col.dtype
    elif gpd_GeometryDtype and isinstance(col.dtype, gpd_GeometryDtype):
        return col.dtype
    dshape = datashape.CType.from_numpy_dtype(col.dtype)
    dshape = datashape.string if dshape == datashape.object_ else dshape
    if dshape in (datashape.string, datashape.datetime_):
        return datashape.Option(dshape)
    return dshape