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
def add_legend_data(self, ax, func, common_kws=None, attrs=None, semantic_kws=None):
    """Add labeled artists to represent the different plot semantics."""
    verbosity = self.legend
    if isinstance(verbosity, str) and verbosity not in ['auto', 'brief', 'full']:
        err = "`legend` must be 'auto', 'brief', 'full', or a boolean."
        raise ValueError(err)
    elif verbosity is True:
        verbosity = 'auto'
    keys = []
    legend_kws = {}
    common_kws = {} if common_kws is None else common_kws.copy()
    semantic_kws = {} if semantic_kws is None else semantic_kws.copy()
    titles = {title for title in (self.variables.get(v, None) for v in ['hue', 'size', 'style']) if title is not None}
    title = '' if len(titles) != 1 else titles.pop()
    title_kws = dict(visible=False, color='w', s=0, linewidth=0, marker='', dashes='')

    def update(var_name, val_name, **kws):
        key = (var_name, val_name)
        if key in legend_kws:
            legend_kws[key].update(**kws)
        else:
            keys.append(key)
            legend_kws[key] = dict(**kws)
    if attrs is None:
        attrs = {'hue': 'color', 'size': ['linewidth', 's'], 'style': None}
    for var, names in attrs.items():
        self._update_legend_data(update, var, verbosity, title, title_kws, names, semantic_kws.get(var))
    legend_data = {}
    legend_order = []
    if common_kws.get('color', False) is None:
        common_kws.pop('color')
    for key in keys:
        _, label = key
        kws = legend_kws[key]
        level_kws = {}
        use_attrs = [*self._legend_attributes, *common_kws, *[attr for var_attrs in semantic_kws.values() for attr in var_attrs]]
        for attr in use_attrs:
            if attr in kws:
                level_kws[attr] = kws[attr]
        artist = func(label=label, **{'color': '.2', **common_kws, **level_kws})
        if _version_predates(mpl, '3.5.0'):
            if isinstance(artist, mpl.lines.Line2D):
                ax.add_line(artist)
            elif isinstance(artist, mpl.patches.Patch):
                ax.add_patch(artist)
            elif isinstance(artist, mpl.collections.Collection):
                ax.add_collection(artist)
        else:
            ax.add_artist(artist)
        legend_data[key] = artist
        legend_order.append(key)
    self.legend_title = title
    self.legend_data = legend_data
    self.legend_order = legend_order