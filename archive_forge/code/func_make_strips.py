from __future__ import annotations
import typing
import numpy as np
import pandas as pd
from .._utils import add_margins, cross_join, join_keys, match, ninteraction
from ..exceptions import PlotnineError
from .facet import (
from .strips import Strips, strip
def make_strips(self, layout_info: layout_details, ax: Axes) -> Strips:
    lst = []
    if layout_info.is_top and self.cols:
        s = strip(self.cols, layout_info, self, ax, 'top')
        lst.append(s)
    if layout_info.is_right and self.rows:
        s = strip(self.rows, layout_info, self, ax, 'right')
        lst.append(s)
    return Strips(lst)