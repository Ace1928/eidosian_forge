from __future__ import annotations
import typing
from contextlib import suppress
import numpy as np
from .._utils import match
from ..exceptions import PlotnineError
from ..iapi import labels_view, layout_details, pos_scales
def reset_position_scales(self):
    """
        Reset x and y scales
        """
    if not self.facet.shrink:
        return
    with suppress(AttributeError):
        self.panel_scales_x.reset()
    with suppress(AttributeError):
        self.panel_scales_y.reset()