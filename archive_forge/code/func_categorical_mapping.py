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
def categorical_mapping(self, data, sizes, order):
    levels = categorical_order(data, order)
    if isinstance(sizes, dict):
        missing = set(levels) - set(sizes)
        if any(missing):
            err = f'Missing sizes for the following levels: {missing}'
            raise ValueError(err)
        lookup_table = sizes.copy()
    elif isinstance(sizes, list):
        sizes = self._check_list_length(levels, sizes, 'sizes')
        lookup_table = dict(zip(levels, sizes))
    else:
        if isinstance(sizes, tuple):
            if len(sizes) != 2:
                err = 'A `sizes` tuple must have only 2 values'
                raise ValueError(err)
        elif sizes is not None:
            err = f'Value for `sizes` not understood: {sizes}'
            raise ValueError(err)
        else:
            sizes = self.plotter._default_size_range
        sizes = np.linspace(*sizes, len(levels))[::-1]
        lookup_table = dict(zip(levels, sizes))
    return (levels, lookup_table)