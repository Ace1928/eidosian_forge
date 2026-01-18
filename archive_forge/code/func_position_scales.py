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
def position_scales(self) -> Scales:
    """
        Return a list of the position scales that are present
        """
    l = [s for s in self if 'x' in s.aesthetics or 'y' in s.aesthetics]
    return Scales(l)