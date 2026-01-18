import sys
from importlib.util import find_spec
import numpy as np
import pandas as pd
from ..core import Dataset, NdOverlay, util
from ..streams import Lasso, Selection1D, SelectionXY
from ..util.transform import dim
from .annotation import HSpan, VSpan
def spatial_select_columnar(xvals, yvals, geometry, geom_method=None):
    if 'cudf' in sys.modules:
        import cudf
        if isinstance(xvals, cudf.Series):
            xvals = xvals.values.astype('float')
            yvals = yvals.values.astype('float')
            try:
                import cuspatial
                result = cuspatial.point_in_polygon(xvals, yvals, cudf.Series([0], index=['selection']), [0], geometry[:, 0], geometry[:, 1])
                return result.values
            except ImportError:
                xvals = np.asarray(xvals)
                yvals = np.asarray(yvals)
    if 'dask' in sys.modules:
        import dask.dataframe as dd
        if isinstance(xvals, dd.Series):
            try:
                xvals.name = 'xvals'
                yvals.name = 'yvals'
                df = xvals.to_frame().join(yvals)
                return df.map_partitions(lambda df, geometry: spatial_select_columnar(df.xvals, df.yvals, geometry), geometry, meta=pd.Series(dtype=bool))
            except Exception:
                xvals = np.asarray(xvals)
                yvals = np.asarray(yvals)
    x0, x1 = (geometry[:, 0].min(), geometry[:, 0].max())
    y0, y1 = (geometry[:, 1].min(), geometry[:, 1].max())
    sel_mask = (xvals >= x0) & (xvals <= x1) & (yvals >= y0) & (yvals <= y1)
    masked_xvals = xvals[sel_mask]
    masked_yvals = yvals[sel_mask]
    if geom_method is None:
        if find_spec('spatialpandas') is not None:
            geom_method = 'spatialpandas'
        elif find_spec('shapely') is not None:
            geom_method = 'shapely'
        else:
            msg = 'Lasso selection on tabular data requires either spatialpandas or shapely to be available.'
            raise ImportError(msg) from None
    geom_function = {'spatialpandas': _mask_spatialpandas, 'shapely': _mask_shapely}[geom_method]
    geom_mask = geom_function(masked_xvals, masked_yvals, geometry)
    if isinstance(xvals, pd.Series):
        sel_mask[sel_mask.index[np.where(sel_mask)[0]]] = geom_mask
    else:
        sel_mask[np.where(sel_mask)[0]] = geom_mask
    return sel_mask