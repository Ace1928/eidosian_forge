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
def get_minor_breaks(self, major: ScaleContinuousBreaks, limits: Optional[ScaleContinuousLimits]=None) -> ScaleContinuousBreaks:
    """
        Return minor breaks
        """
    if limits is None:
        limits = self.limits
    if self.minor_breaks is False or self.minor_breaks is None:
        minor_breaks = []
    elif self.minor_breaks is True:
        minor_breaks: ScaleContinuousBreaks = self.trans.minor_breaks(major, limits)
    elif isinstance(self.minor_breaks, int):
        minor_breaks: ScaleContinuousBreaks = self.trans.minor_breaks(major, limits, self.minor_breaks)
    elif callable(self.minor_breaks):
        breaks = self.minor_breaks(self.inverse(limits))
        _major = set(major)
        minor = self.transform(breaks)
        minor_breaks = [x for x in minor if x not in _major]
    else:
        minor_breaks = self.transform(self.minor_breaks)
    return minor_breaks