from __future__ import annotations
import typing
from contextlib import suppress
from warnings import warn
import numpy as np
from .._utils import order_as_data_mapping, to_rgba
from ..doctools import document
from ..exceptions import PlotnineError, PlotnineWarning
from ..positions import position_nudge
from .geom import geom
def check_adjust_text():
    try:
        pass
    except ImportError as err:
        msg = 'To use adjust_text you must install the adjustText package.'
        raise PlotnineError(msg) from err