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
@property
def comp_data(self):
    """Dataframe with numeric x and y, after unit conversion and log scaling."""
    if not hasattr(self, 'ax'):
        return self.plot_data
    if not hasattr(self, '_comp_data'):
        comp_data = self.plot_data.copy(deep=False).drop(['x', 'y'], axis=1, errors='ignore')
        for var in 'yx':
            if var not in self.variables:
                continue
            parts = []
            grouped = self.plot_data[var].groupby(self.converters[var], sort=False)
            for converter, orig in grouped:
                orig = orig.mask(orig.isin([np.inf, -np.inf]), np.nan)
                orig = orig.dropna()
                if var in self.var_levels:
                    orig = orig[orig.isin(self.var_levels[var])]
                comp = pd.to_numeric(converter.convert_units(orig)).astype(float)
                transform = converter.get_transform().transform
                parts.append(pd.Series(transform(comp), orig.index, name=orig.name))
            if parts:
                comp_col = pd.concat(parts)
            else:
                comp_col = pd.Series(dtype=float, name=var)
            comp_data.insert(0, var, comp_col)
        self._comp_data = comp_data
    return self._comp_data