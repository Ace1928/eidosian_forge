from __future__ import annotations
import re
from copy import copy
from collections.abc import Sequence
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Tuple, Optional, ClassVar
import numpy as np
import matplotlib as mpl
from matplotlib.ticker import (
from matplotlib.dates import (
from matplotlib.axis import Axis
from matplotlib.scale import ScaleBase
from pandas import Series
from seaborn._core.rules import categorical_order
from seaborn._core.typing import Default, default
from typing import TYPE_CHECKING
def na_safe_cast(x):
    if np.isscalar(x):
        return float(bool(x))
    else:
        if hasattr(x, 'notna'):
            use = x.notna().to_numpy()
        else:
            use = np.isfinite(x)
        out = np.full(len(x), np.nan, dtype=float)
        out[use] = x[use].astype(bool).astype(float)
        return out