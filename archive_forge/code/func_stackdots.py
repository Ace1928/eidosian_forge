from __future__ import annotations
import typing
from warnings import warn
import numpy as np
from .._utils import groupby_apply, resolution, to_rgba
from ..doctools import document
from ..exceptions import PlotnineWarning
from .geom import geom
from .geom_path import geom_path
def stackdots(a: FloatSeries) -> FloatSeries:
    return a - 1 - np.floor(np.max(a - 1) / 2)