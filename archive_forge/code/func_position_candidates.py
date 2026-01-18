from collections import namedtuple
from textwrap import dedent
import warnings
from colorsys import rgb_to_hls
from functools import partial
import numpy as np
import pandas as pd
import matplotlib as mpl
from matplotlib.cbook import normalize_kwargs
from matplotlib.collections import PatchCollection
from matplotlib.markers import MarkerStyle
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
from seaborn._core.typing import default, deprecated
from seaborn._base import VectorPlotter, infer_orient, categorical_order
from seaborn._stats.density import KDE
from seaborn import utils
from seaborn.utils import (
from seaborn._compat import groupby_apply_include_groups
from seaborn._statistics import (
from seaborn.palettes import light_palette
from seaborn.axisgrid import FacetGrid, _facet_docs
def position_candidates(self, xyr_i, neighbors):
    """Return a list of coordinates that might be valid by adjusting x."""
    candidates = [xyr_i]
    x_i, y_i, r_i = xyr_i
    left_first = True
    for x_j, y_j, r_j in neighbors:
        dy = y_i - y_j
        dx = np.sqrt(max((r_i + r_j) ** 2 - dy ** 2, 0)) * 1.05
        cl, cr = ((x_j - dx, y_i, r_i), (x_j + dx, y_i, r_i))
        if left_first:
            new_candidates = [cl, cr]
        else:
            new_candidates = [cr, cl]
        candidates.extend(new_candidates)
        left_first = not left_first
    return np.array(candidates)