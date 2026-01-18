import os
import inspect
import warnings
import colorsys
from contextlib import contextmanager
from urllib.request import urlopen, urlretrieve
from types import ModuleType
import numpy as np
import pandas as pd
import matplotlib as mpl
from matplotlib.colors import to_rgb
import matplotlib.pyplot as plt
from matplotlib.cbook import normalize_kwargs
from seaborn._core.typing import deprecated
from seaborn.external.version import Version
from seaborn.external.appdirs import user_cache_dir
def move_legend(obj, loc, **kwargs):
    """
    Recreate a plot's legend at a new location.

    The name is a slight misnomer. Matplotlib legends do not expose public
    control over their position parameters. So this function creates a new legend,
    copying over the data from the original object, which is then removed.

    Parameters
    ----------
    obj : the object with the plot
        This argument can be either a seaborn or matplotlib object:

        - :class:`seaborn.FacetGrid` or :class:`seaborn.PairGrid`
        - :class:`matplotlib.axes.Axes` or :class:`matplotlib.figure.Figure`

    loc : str or int
        Location argument, as in :meth:`matplotlib.axes.Axes.legend`.

    kwargs
        Other keyword arguments are passed to :meth:`matplotlib.axes.Axes.legend`.

    Examples
    --------

    .. include:: ../docstrings/move_legend.rst

    """
    from seaborn.axisgrid import Grid
    if isinstance(obj, Grid):
        old_legend = obj.legend
        legend_func = obj.figure.legend
    elif isinstance(obj, mpl.axes.Axes):
        old_legend = obj.legend_
        legend_func = obj.legend
    elif isinstance(obj, mpl.figure.Figure):
        if obj.legends:
            old_legend = obj.legends[-1]
        else:
            old_legend = None
        legend_func = obj.legend
    else:
        err = '`obj` must be a seaborn Grid or matplotlib Axes or Figure instance.'
        raise TypeError(err)
    if old_legend is None:
        err = f'{obj} has no legend attached.'
        raise ValueError(err)
    from seaborn._compat import get_legend_handles
    handles = get_legend_handles(old_legend)
    labels = [t.get_text() for t in old_legend.get_texts()]
    if (new_labels := kwargs.pop('labels', None)) is not None:
        if len(new_labels) != len(labels):
            err = 'Length of new labels does not match existing legend.'
            raise ValueError(err)
        labels = new_labels
    legend_kws = inspect.signature(mpl.legend.Legend).parameters
    props = {k: v for k, v in old_legend.properties().items() if k in legend_kws}
    props.pop('bbox_to_anchor')
    title = props.pop('title')
    if 'title' in kwargs:
        title.set_text(kwargs.pop('title'))
    title_kwargs = {k: v for k, v in kwargs.items() if k.startswith('title_')}
    for key, val in title_kwargs.items():
        title.set(**{key[6:]: val})
        kwargs.pop(key)
    kwargs.setdefault('frameon', old_legend.legendPatch.get_visible())
    props.update(kwargs)
    old_legend.remove()
    new_legend = legend_func(handles, labels, loc=loc, **props)
    new_legend.set_title(title.get_text(), title.get_fontproperties())
    if isinstance(obj, Grid):
        obj._legend = new_legend