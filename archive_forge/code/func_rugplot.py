from numbers import Number
from functools import partial
import math
import textwrap
import warnings
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.transforms as tx
from matplotlib.cbook import normalize_kwargs
from matplotlib.colors import to_rgba
from matplotlib.collections import LineCollection
from ._base import VectorPlotter
from ._statistics import ECDF, Histogram, KDE
from ._stats.counting import Hist
from .axisgrid import (
from .utils import (
from .palettes import color_palette
from .external import husl
from .external.kde import gaussian_kde
from ._docstrings import (
important parameter. Misspecification of the bandwidth can produce a
def rugplot(data=None, *, x=None, y=None, hue=None, height=0.025, expand_margins=True, palette=None, hue_order=None, hue_norm=None, legend=True, ax=None, **kwargs):
    a = kwargs.pop('a', None)
    axis = kwargs.pop('axis', None)
    if a is not None:
        data = a
        msg = textwrap.dedent('\n\n        The `a` parameter has been replaced; use `x`, `y`, and/or `data` instead.\n        Please update your code; This will become an error in seaborn v0.14.0.\n        ')
        warnings.warn(msg, UserWarning, stacklevel=2)
    if axis is not None:
        if axis == 'x':
            x = data
        elif axis == 'y':
            y = data
        data = None
        msg = textwrap.dedent(f'\n\n        The `axis` parameter has been deprecated; use the `{axis}` parameter instead.\n        Please update your code; this will become an error in seaborn v0.14.0.\n        ')
        warnings.warn(msg, UserWarning, stacklevel=2)
    vertical = kwargs.pop('vertical', None)
    if vertical is not None:
        if vertical:
            action_taken = 'assigning data to `y`.'
            if x is None:
                data, y = (y, data)
            else:
                x, y = (y, x)
        else:
            action_taken = 'assigning data to `x`.'
        msg = textwrap.dedent(f'\n\n        The `vertical` parameter is deprecated; {action_taken}\n        This will become an error in seaborn v0.14.0; please update your code.\n        ')
        warnings.warn(msg, UserWarning, stacklevel=2)
    p = _DistributionPlotter(data=data, variables=dict(x=x, y=y, hue=hue))
    p.map_hue(palette=palette, order=hue_order, norm=hue_norm)
    if ax is None:
        ax = plt.gca()
    p._attach(ax)
    color = kwargs.pop('color', kwargs.pop('c', None))
    kwargs['color'] = _default_color(ax.plot, hue, color, kwargs)
    if not p.has_xy_data:
        return ax
    p.plot_rug(height, expand_margins, legend, **kwargs)
    return ax