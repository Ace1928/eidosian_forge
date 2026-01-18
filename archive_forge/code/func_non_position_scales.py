from __future__ import annotations
import itertools
import typing
from contextlib import suppress
from typing import List
from warnings import warn
import numpy as np
import pandas.api.types as pdtypes
from .._utils import array_kind
from .._utils.registry import Registry
from ..exceptions import PlotnineError, PlotnineWarning
from ..mapping.aes import aes_to_scale
from .scale import scale
def non_position_scales(self) -> Scales:
    """
        Return a list of any non-position scales
        """
    l = [s for s in self if 'x' not in s.aesthetics and 'y' not in s.aesthetics]
    return Scales(l)