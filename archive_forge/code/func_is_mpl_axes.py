from __future__ import annotations
import datetime as dt
import os
import sys
from typing import Any, Iterable
import numpy as np
import param
def is_mpl_axes(obj) -> bool:
    if 'matplotlib' not in sys.modules:
        return False
    from matplotlib.axes import Axes
    return isinstance(obj, Axes)