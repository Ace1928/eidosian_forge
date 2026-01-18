from __future__ import annotations
import typing
from contextlib import suppress
from warnings import warn
import numpy as np
import pandas as pd
from mizani.bounds import censor, expand_range_distinct, rescale, zero_range
from .._utils import match
from ..doctools import document
from ..exceptions import PlotnineError, PlotnineWarning
from ..iapi import range_view, scale_view
from ._expand import expand_range
from .range import RangeContinuous
from .scale import scale
def expand_limits(self, limits: ScaleContinuousLimits, expand: TupleFloat2 | TupleFloat4, coord_limits: CoordRange | None, trans: trans) -> range_view:
    """
        Calculate the final range in coordinate space
        """
    if coord_limits is not None:
        c0, c1 = coord_limits
        limits = (limits[0] if c0 is None else c0, limits[1] if c1 is None else c1)
    return expand_range(limits, expand, trans)