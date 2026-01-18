from contextlib import ExitStack
import functools
import math
from numbers import Integral
import numpy as np
from numpy import ma
import matplotlib as mpl
from matplotlib import _api, _docstring
from matplotlib.backend_bases import MouseButton
from matplotlib.lines import Line2D
from matplotlib.path import Path
from matplotlib.text import Text
import matplotlib.ticker as ticker
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.collections as mcoll
import matplotlib.font_manager as font_manager
import matplotlib.cbook as cbook
import matplotlib.patches as mpatches
import matplotlib.transforms as mtransforms
def locate_label(self, linecontour, labelwidth):
    """
        Find good place to draw a label (relatively flat part of the contour).
        """
    ctr_size = len(linecontour)
    n_blocks = int(np.ceil(ctr_size / labelwidth)) if labelwidth > 1 else 1
    block_size = ctr_size if n_blocks == 1 else int(labelwidth)
    xx = np.resize(linecontour[:, 0], (n_blocks, block_size))
    yy = np.resize(linecontour[:, 1], (n_blocks, block_size))
    yfirst = yy[:, :1]
    ylast = yy[:, -1:]
    xfirst = xx[:, :1]
    xlast = xx[:, -1:]
    s = (yfirst - yy) * (xlast - xfirst) - (xfirst - xx) * (ylast - yfirst)
    l = np.hypot(xlast - xfirst, ylast - yfirst)
    with np.errstate(divide='ignore', invalid='ignore'):
        distances = (abs(s) / l).sum(axis=-1)
    hbsize = block_size // 2
    adist = np.argsort(distances)
    for idx in np.append(adist, adist[0]):
        x, y = (xx[idx, hbsize], yy[idx, hbsize])
        if not self.too_close(x, y, labelwidth):
            break
    return (x, y, (idx * block_size + hbsize) % ctr_size)