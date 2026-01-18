from __future__ import annotations
import typing
from contextlib import suppress
from warnings import warn
import numpy as np
from .._utils import order_as_data_mapping, to_rgba
from ..doctools import document
from ..exceptions import PlotnineError, PlotnineWarning
from ..positions import position_nudge
from .geom import geom
@staticmethod
def legend_key_size(data: pd.Series[Any], min_size: TupleInt2, lyr: layer) -> TupleInt2:
    w, h = min_size
    _w = _h = data['size']
    if data['color'] is not None:
        w = max(w, _w)
        h = max(h, _h)
    return (w, h)