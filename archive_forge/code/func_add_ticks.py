from __future__ import annotations
import hashlib
from dataclasses import dataclass, field
from functools import cached_property
from types import SimpleNamespace as NS
from typing import TYPE_CHECKING, cast
from warnings import warn
import numpy as np
import pandas as pd
from mizani.bounds import rescale
from .._utils import get_opposite_side
from ..exceptions import PlotnineError, PlotnineWarning
from ..mapping.aes import rename_aesthetics
from ..scales.scale_continuous import scale_continuous
from .guide import GuideElements, guide
def add_ticks(auxbox, locations, elements) -> LineCollection:
    """
    Add ticks to colorbar
    """
    from matplotlib.collections import LineCollection
    segments = []
    l = elements.ticks_length
    tick_stops = np.array([0.0, l, 1 - l, 1]) * elements.key_width
    if elements.is_vertical:
        x1, x2, x3, x4 = tick_stops
        for y in locations:
            segments.extend([((x1, y), (x2, y)), ((x3, y), (x4, y))])
    else:
        y1, y2, y3, y4 = tick_stops
        for x in locations:
            segments.extend([((x, y1), (x, y2)), ((x, y3), (x, y4))])
    coll = LineCollection(segments)
    auxbox.add_artist(coll)
    return coll