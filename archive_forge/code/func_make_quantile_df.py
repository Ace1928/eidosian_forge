from __future__ import annotations
import typing
import numpy as np
import pandas as pd
from .._utils import groupby_apply, interleave, resolution
from ..doctools import document
from .geom import geom
from .geom_path import geom_path
from .geom_polygon import geom_polygon
def make_quantile_df(data: pd.DataFrame, draw_quantiles: FloatArray) -> pd.DataFrame:
    """
    Return a dataframe with info needed to draw quantile segments
    """
    from scipy.interpolate import interp1d
    dens = data['density'].cumsum() / data['density'].sum()
    ecdf = interp1d(dens, data['y'], assume_sorted=True)
    ys = ecdf(draw_quantiles)
    violin_xminvs = interp1d(data['y'], data['xminv'])(ys)
    violin_xmaxvs = interp1d(data['y'], data['xmaxv'])(ys)
    data = pd.DataFrame({'x': interleave(violin_xminvs, violin_xmaxvs), 'y': np.repeat(ys, 2), 'group': np.repeat(np.arange(1, len(ys) + 1), 2)})
    return data