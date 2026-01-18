import bisect
import re
import traceback
import warnings
from collections import defaultdict, namedtuple
import numpy as np
import param
from packaging.version import Version
from ..core import (
from ..core.ndmapping import item_check
from ..core.operation import Operation
from ..core.options import CallbackError, Cycle
from ..core.spaces import get_nested_streams
from ..core.util import (
from ..element import Points
from ..streams import LinkedStream, Params
from ..util.transform import dim
def color_intervals(colors, levels, clip=None, N=255):
    """
    Maps the supplied colors into bins defined by the supplied levels.
    If a clip tuple is defined the bins are clipped to the defined
    range otherwise the range is computed from the levels and returned.

    Arguments
    ---------
    colors: list
      List of colors (usually hex string or named colors)
    levels: list or array_like
      Levels specifying the bins to map the colors to
    clip: tuple (optional)
      Lower and upper limits of the color range
    N: int
      Number of discrete colors to map the range onto

    Returns
    -------
    cmap: list
      List of colors
    clip: tuple
      Lower and upper bounds of the color range
    """
    if len(colors) != len(levels) - 1:
        raise ValueError('The number of colors in the colormap must match the intervals defined in the color_levels, expected %d colors found %d.' % (N, len(colors)))
    intervals = np.diff(levels)
    cmin, cmax = (min(levels), max(levels))
    interval = cmax - cmin
    cmap = []
    for intv, c in zip(intervals, colors):
        cmap += [c] * int(round(N * (intv / interval)))
    if clip is not None:
        clmin, clmax = clip
        lidx = int(round(N * ((clmin - cmin) / interval)))
        uidx = int(round(N * ((cmax - clmax) / interval)))
        uidx = N - uidx
        if lidx == uidx:
            uidx = lidx + 1
        cmap = cmap[lidx:uidx]
        if clmin == clmax:
            idx = np.argmin(np.abs(np.array(levels) - clmin))
            clip = levels[idx:idx + 2] if len(levels) > idx + 2 else levels[idx - 1:idx + 1]
    return (cmap, clip)