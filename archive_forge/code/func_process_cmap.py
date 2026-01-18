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
def process_cmap(cmap, ncolors=None, provider=None, categorical=False):
    """
    Convert valid colormap specifications to a list of colors.
    """
    providers_checked = 'matplotlib, bokeh, or colorcet' if provider is None else provider
    if isinstance(cmap, Cycle):
        palette = [rgb2hex(c) if isinstance(c, tuple) else c for c in cmap.values]
    elif isinstance(cmap, tuple):
        palette = list(cmap)
    elif isinstance(cmap, list):
        palette = cmap
    elif isinstance(cmap, str):
        mpl_cmaps = _list_cmaps('matplotlib')
        bk_cmaps = _list_cmaps('bokeh')
        cet_cmaps = _list_cmaps('colorcet')
        if provider == 'matplotlib' or (provider is None and (cmap in mpl_cmaps or cmap.lower() in mpl_cmaps)):
            palette = mplcmap_to_palette(cmap, ncolors, categorical)
        elif provider == 'bokeh' or (provider is None and (cmap in bk_cmaps or cmap.capitalize() in bk_cmaps)):
            palette = bokeh_palette_to_palette(cmap, ncolors, categorical)
        elif provider == 'colorcet' or (provider is None and cmap in cet_cmaps):
            palette = colorcet_cmap_to_palette(cmap, ncolors, categorical)
        else:
            raise ValueError(f'Supplied cmap {cmap} not found among {providers_checked} colormaps.')
    else:
        try:
            palette = mplcmap_to_palette(cmap, ncolors)
        except Exception:
            palette = None
    if not isinstance(palette, list):
        raise TypeError(f'cmap argument {cmap} expects a list, Cycle or valid {providers_checked} colormap or palette.')
    if ncolors and len(palette) != ncolors:
        return [palette[i % len(palette)] for i in range(ncolors)]
    return palette