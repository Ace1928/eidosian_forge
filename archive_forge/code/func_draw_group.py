from __future__ import annotations
import typing
from .._utils import SIZE_FACTOR, to_rgba
from ..coords import coord_flip
from ..doctools import document
from ..exceptions import PlotnineError
from .geom import geom
from .geom_path import geom_path
from .geom_polygon import geom_polygon
@staticmethod
def draw_group(data: pd.DataFrame, panel_params: panel_view, coord: coord, ax: Axes, **params: Any):
    _x = 'y' if isinstance(coord, coord_flip) else 'x'
    data = coord.transform(data, panel_params, munch=True)
    data = data.sort_values(by=['group', _x], kind='mergesort')
    units = ['alpha', 'color', 'fill', 'linetype', 'size']
    if len(data[units].drop_duplicates()) > 1:
        msg = 'Aesthetics cannot vary within a ribbon.'
        raise PlotnineError(msg)
    for _, udata in data.groupby(units, dropna=False):
        udata.reset_index(inplace=True, drop=True)
        geom_ribbon.draw_unit(udata, panel_params, coord, ax, **params)