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
def mplcmap_to_palette(cmap, ncolors=None, categorical=False):
    """
    Converts a matplotlib colormap to palette of RGB hex strings."
    """
    import matplotlib as mpl
    from matplotlib.colors import Colormap, ListedColormap
    ncolors = ncolors or 256
    if not isinstance(cmap, Colormap):
        if cmap.startswith('Category'):
            cmap = cmap.replace('Category', 'tab')
        if Version(mpl.__version__) < Version('3.5'):
            from matplotlib import cm
            try:
                cmap = cm.get_cmap(cmap)
            except Exception:
                cmap = cm.get_cmap(cmap.lower())
        else:
            from matplotlib import colormaps
            cmap = colormaps.get(cmap, colormaps.get(cmap.lower()))
    if isinstance(cmap, ListedColormap):
        if categorical:
            palette = [rgb2hex(cmap.colors[i % cmap.N]) for i in range(ncolors)]
            return palette
        elif cmap.N > ncolors:
            palette = [rgb2hex(c) for c in cmap(np.arange(cmap.N))]
            if len(palette) != ncolors:
                palette = [palette[int(v)] for v in np.linspace(0, len(palette) - 1, ncolors)]
            return palette
    return [rgb2hex(c) for c in cmap(np.linspace(0, 1, ncolors))]