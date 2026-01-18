from __future__ import annotations
import typing
import numpy as np
import pandas as pd
from .._utils import SIZE_FACTOR, to_rgba
from ..doctools import document
from ..exceptions import PlotnineError
from .geom import geom
from .geom_point import geom_point
from .geom_polygon import geom_polygon
def ccw_coords(ring: LinearRing) -> npt.NDArray[Any]:
    """
        Return Counter Clockwise array coordinates

        Parameters
        ----------
        ring: shapely.geometry.polygon.LinearRing
            LinearRing

        Returns
        -------
        out: ndarray
            (n x 2) array of coordinate points.
        """
    if ring.is_ccw:
        return np.asarray(ring.coords)[:, :2]
    return np.asarray(ring.coords)[:, :2][::-1]