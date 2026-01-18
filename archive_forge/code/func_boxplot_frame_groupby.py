from __future__ import annotations
from typing import (
import warnings
from matplotlib.artist import setp
import numpy as np
from pandas._libs import lib
from pandas.util._decorators import cache_readonly
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.common import is_dict_like
from pandas.core.dtypes.generic import ABCSeries
from pandas.core.dtypes.missing import remove_na_arraylike
import pandas as pd
import pandas.core.common as com
from pandas.io.formats.printing import pprint_thing
from pandas.plotting._matplotlib.core import (
from pandas.plotting._matplotlib.groupby import create_iter_data_given_by
from pandas.plotting._matplotlib.style import get_standard_colors
from pandas.plotting._matplotlib.tools import (
def boxplot_frame_groupby(grouped, subplots: bool=True, column=None, fontsize: int | None=None, rot: int=0, grid: bool=True, ax=None, figsize: tuple[float, float] | None=None, layout=None, sharex: bool=False, sharey: bool=True, **kwds):
    if subplots is True:
        naxes = len(grouped)
        fig, axes = create_subplots(naxes=naxes, squeeze=False, ax=ax, sharex=sharex, sharey=sharey, figsize=figsize, layout=layout)
        axes = flatten_axes(axes)
        ret = pd.Series(dtype=object)
        for (key, group), ax in zip(grouped, axes):
            d = group.boxplot(ax=ax, column=column, fontsize=fontsize, rot=rot, grid=grid, **kwds)
            ax.set_title(pprint_thing(key))
            ret.loc[key] = d
        maybe_adjust_figure(fig, bottom=0.15, top=0.9, left=0.1, right=0.9, wspace=0.2)
    else:
        keys, frames = zip(*grouped)
        if grouped.axis == 0:
            df = pd.concat(frames, keys=keys, axis=1)
        elif len(frames) > 1:
            df = frames[0].join(frames[1:])
        else:
            df = frames[0]
        if column is not None:
            column = com.convert_to_list_like(column)
            multi_key = pd.MultiIndex.from_product([keys, column])
            column = list(multi_key.values)
        ret = df.boxplot(column=column, fontsize=fontsize, rot=rot, grid=grid, ax=ax, figsize=figsize, layout=layout, **kwds)
    return ret