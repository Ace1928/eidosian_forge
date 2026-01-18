from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass, fields
from functools import cached_property
from typing import TYPE_CHECKING, Literal, cast
from warnings import warn
import numpy as np
import pandas as pd
from .._utils import ensure_xy_location
from .._utils.registry import Registry
from ..exceptions import PlotnineError, PlotnineWarning
from ..iapi import (
from ..mapping.aes import rename_aesthetics
from .guide import guide
@cached_property
def justification(self) -> legend_justifications_view:
    if self.position == 'none':
        return legend_justifications_view()
    dim_lookup = {'left': 1, 'right': 1, 'top': 0, 'bottom': 0}

    def _lrtb(pos):
        just = self.theme.getp(f'legend_justification_{pos}')
        idx = dim_lookup[pos]
        if just is None:
            just = (0.5, 0.5)
        elif just in VALID_JUSTIFICATION_WORDS:
            just = ensure_xy_location(just)
        elif isinstance(just, (float, int)):
            just = (just, just)
        return just[idx]

    def _inside():
        just = self.theme.getp('legend_justification_inside')
        if just is None:
            return None
        return ensure_xy_location(just)
    return legend_justifications_view(left=_lrtb('left'), right=_lrtb('right'), top=_lrtb('top'), bottom=_lrtb('bottom'), inside=_inside())