from __future__ import annotations
from typing import (
import numpy as np
from pandas.core.dtypes.common import (
from pandas.core.dtypes.generic import (
from pandas.core.dtypes.missing import (
from pandas.io.formats.printing import pprint_thing
from pandas.plotting._matplotlib.core import (
from pandas.plotting._matplotlib.groupby import (
from pandas.plotting._matplotlib.misc import unpack_single_str_list
from pandas.plotting._matplotlib.tools import (
def hist_frame(data: DataFrame, column=None, by=None, grid: bool=True, xlabelsize: int | None=None, xrot=None, ylabelsize: int | None=None, yrot=None, ax=None, sharex: bool=False, sharey: bool=False, figsize: tuple[float, float] | None=None, layout=None, bins: int=10, legend: bool=False, **kwds):
    if legend and 'label' in kwds:
        raise ValueError('Cannot use both legend and label')
    if by is not None:
        axes = _grouped_hist(data, column=column, by=by, ax=ax, grid=grid, figsize=figsize, sharex=sharex, sharey=sharey, layout=layout, bins=bins, xlabelsize=xlabelsize, xrot=xrot, ylabelsize=ylabelsize, yrot=yrot, legend=legend, **kwds)
        return axes
    if column is not None:
        if not isinstance(column, (list, np.ndarray, ABCIndex)):
            column = [column]
        data = data[column]
    data = data.select_dtypes(include=(np.number, 'datetime64', 'datetimetz'), exclude='timedelta')
    naxes = len(data.columns)
    if naxes == 0:
        raise ValueError('hist method requires numerical or datetime columns, nothing to plot.')
    fig, axes = create_subplots(naxes=naxes, ax=ax, squeeze=False, sharex=sharex, sharey=sharey, figsize=figsize, layout=layout)
    _axes = flatten_axes(axes)
    can_set_label = 'label' not in kwds
    for i, col in enumerate(data.columns):
        ax = _axes[i]
        if legend and can_set_label:
            kwds['label'] = col
        ax.hist(data[col].dropna().values, bins=bins, **kwds)
        ax.set_title(col)
        ax.grid(grid)
        if legend:
            ax.legend()
    set_ticks_props(axes, xlabelsize=xlabelsize, xrot=xrot, ylabelsize=ylabelsize, yrot=yrot)
    maybe_adjust_figure(fig, wspace=0.3, hspace=0.3)
    return axes