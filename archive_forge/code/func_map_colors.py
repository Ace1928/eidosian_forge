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
def map_colors(arr, crange, cmap, hex=True):
    """
    Maps an array of values to RGB hex strings, given
    a color range and colormap.
    """
    if isinstance(crange, arraylike_types):
        xsorted = np.argsort(crange)
        ypos = np.searchsorted(crange, arr)
        arr = xsorted[ypos]
    else:
        if isinstance(crange, tuple):
            cmin, cmax = crange
        else:
            cmin, cmax = (np.nanmin(arr), np.nanmax(arr))
        arr = (arr - cmin) / (cmax - cmin)
        arr = np.ma.array(arr, mask=np.logical_not(np.isfinite(arr)))
    arr = cmap(arr)
    if hex:
        return rgb2hex(arr)
    else:
        return arr