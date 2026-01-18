from __future__ import annotations
import io
import os
import re
import inspect
import itertools
import textwrap
from contextlib import contextmanager
from collections import abc
from collections.abc import Callable, Generator
from typing import Any, List, Literal, Optional, cast
from xml.etree import ElementTree
from cycler import cycler
import pandas as pd
from pandas import DataFrame, Series, Index
import matplotlib as mpl
from matplotlib.axes import Axes
from matplotlib.artist import Artist
from matplotlib.figure import Figure
import numpy as np
from PIL import Image
from seaborn._marks.base import Mark
from seaborn._stats.base import Stat
from seaborn._core.data import PlotData
from seaborn._core.moves import Move
from seaborn._core.scales import Scale
from seaborn._core.subplots import Subplots
from seaborn._core.groupby import GroupBy
from seaborn._core.properties import PROPERTIES, Property
from seaborn._core.typing import (
from seaborn._core.exceptions import PlotSpecError
from seaborn._core.rules import categorical_order
from seaborn._compat import get_layout_engine, set_layout_engine
from seaborn.utils import _version_predates
from seaborn.rcmod import axes_style, plotting_context
from seaborn.palettes import color_palette
from typing import TYPE_CHECKING, TypedDict
def split_generator(keep_na=False) -> Generator:
    for view in subplots:
        axes_df = self._filter_subplot_data(df, view)
        axes_df_inf_as_nan = axes_df.copy()
        axes_df_inf_as_nan = axes_df_inf_as_nan.mask(axes_df_inf_as_nan.isin([np.inf, -np.inf]), np.nan)
        if keep_na:
            present = axes_df_inf_as_nan.notna().all(axis=1)
            nulled = {}
            for axis in 'xy':
                if axis in axes_df:
                    nulled[axis] = axes_df[axis].where(present)
            axes_df = axes_df_inf_as_nan.assign(**nulled)
        else:
            axes_df = axes_df_inf_as_nan.dropna()
        subplot_keys = {}
        for dim in ['col', 'row']:
            if view[dim] is not None:
                subplot_keys[dim] = view[dim]
        if not grouping_vars or not any(grouping_keys):
            if not axes_df.empty:
                yield (subplot_keys, axes_df.copy(), view['ax'])
            continue
        grouped_df = axes_df.groupby(grouping_vars, sort=False, as_index=False, observed=False)
        for key in itertools.product(*grouping_keys):
            pd_key = key[0] if len(key) == 1 and _version_predates(pd, '2.2.0') else key
            try:
                df_subset = grouped_df.get_group(pd_key)
            except KeyError:
                df_subset = axes_df.loc[[]]
            if df_subset.empty:
                continue
            sub_vars = dict(zip(grouping_vars, key))
            sub_vars.update(subplot_keys)
            yield (sub_vars, df_subset.copy(), view['ax'])