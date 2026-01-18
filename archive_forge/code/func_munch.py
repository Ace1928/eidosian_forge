from __future__ import annotations
import typing
from copy import copy
import numpy as np
from ..iapi import panel_ranges
def munch(self, data: pd.DataFrame, panel_params: panel_view) -> pd.DataFrame:
    ranges = self.backtransform_range(panel_params)
    x_neginf = np.isneginf(data['x'])
    x_posinf = np.isposinf(data['x'])
    y_neginf = np.isneginf(data['y'])
    y_posinf = np.isposinf(data['y'])
    if x_neginf.any():
        data.loc[x_neginf, 'x'] = ranges.x[0]
    if x_posinf.any():
        data.loc[x_posinf, 'x'] = ranges.x[1]
    if y_neginf.any():
        data.loc[y_neginf, 'y'] = ranges.y[0]
    if y_posinf.any():
        data.loc[y_posinf, 'y'] = ranges.y[1]
    dist = self.distance(data['x'], data['y'], panel_params)
    bool_idx = data['group'].to_numpy()[1:] != data['group'].to_numpy()[:-1]
    dist[bool_idx] = np.nan
    munched = munch_data(data, dist)
    return munched