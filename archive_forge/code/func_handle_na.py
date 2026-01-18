from __future__ import annotations
import typing
from .._utils import SIZE_FACTOR, to_rgba
from ..coords import coord_flip
from ..doctools import document
from ..exceptions import PlotnineError
from .geom import geom
from .geom_path import geom_path
from .geom_polygon import geom_polygon
def handle_na(self, data: pd.DataFrame) -> pd.DataFrame:
    return data