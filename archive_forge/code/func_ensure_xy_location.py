from __future__ import annotations
import inspect
import itertools
import warnings
from collections import defaultdict
from collections.abc import Iterable, Sequence
from contextlib import suppress
from copy import deepcopy
from dataclasses import field
from typing import TYPE_CHECKING, cast, overload
from warnings import warn
import numpy as np
import pandas as pd
from pandas.core.groupby import DataFrameGroupBy
from ..exceptions import PlotnineError, PlotnineWarning
from ..mapping import aes
def ensure_xy_location(loc: SidePosition | Literal['center'] | float | TupleFloat2) -> TupleFloat2:
    """
    Convert input into (x, y) location

    Parameters
    ----------
    loc:
        A specification for a location that can be converted to
        coordinate points on a unit-square. Note that, if the location
        is (x, y) points, the same points are returned.
    """
    if loc in BOX_LOCATIONS:
        return BOX_LOCATIONS[loc]
    elif isinstance(loc, (float, int)):
        return (loc, 0.5)
    elif isinstance(loc, tuple):
        h, v = loc
        if isinstance(h, str) and isinstance(v, str):
            return (BOX_LOCATIONS[h][0], BOX_LOCATIONS[v][1])
        if isinstance(h, (int, float)) and isinstance(v, (int, float)):
            return (h, v)
    raise ValueError(f"Cannot make a location from '{loc}'")