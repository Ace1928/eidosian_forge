from __future__ import annotations
import warnings
import itertools
from copy import copy
from collections import UserString
from collections.abc import Iterable, Sequence, Mapping
from numbers import Number
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib as mpl
from seaborn._core.data import PlotData
from seaborn.palettes import (
from seaborn.utils import (
def numeric_mapping(self, data, sizes, norm):
    if isinstance(sizes, dict):
        levels = list(np.sort(list(sizes)))
        size_values = sizes.values()
        size_range = (min(size_values), max(size_values))
    else:
        levels = list(np.sort(remove_na(data.unique())))
        if isinstance(sizes, tuple):
            if len(sizes) != 2:
                err = 'A `sizes` tuple must have only 2 values'
                raise ValueError(err)
            size_range = sizes
        elif sizes is not None:
            err = f'Value for `sizes` not understood: {sizes}'
            raise ValueError(err)
        else:
            size_range = self.plotter._default_size_range
    if norm is None:
        norm = mpl.colors.Normalize()
    elif isinstance(norm, tuple):
        norm = mpl.colors.Normalize(*norm)
    elif not isinstance(norm, mpl.colors.Normalize):
        err = f'Value for size `norm` parameter not understood: {norm}'
        raise ValueError(err)
    else:
        norm = copy(norm)
    norm.clip = True
    if not norm.scaled():
        norm(levels)
    sizes_scaled = norm(levels)
    if isinstance(sizes, dict):
        lookup_table = sizes
    else:
        lo, hi = size_range
        sizes = lo + sizes_scaled * (hi - lo)
        lookup_table = dict(zip(levels, sizes))
    return (levels, lookup_table, norm, size_range)