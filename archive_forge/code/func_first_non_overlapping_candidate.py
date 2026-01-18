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
def first_non_overlapping_candidate(self, candidates, neighbors):
    """Find the first candidate that does not overlap with the swarm."""
    if len(neighbors) == 0:
        return candidates[0]
    neighbors_x = neighbors[:, 0]
    neighbors_y = neighbors[:, 1]
    neighbors_r = neighbors[:, 2]
    for xyr_i in candidates:
        x_i, y_i, r_i = xyr_i
        dx = neighbors_x - x_i
        dy = neighbors_y - y_i
        sq_distances = np.square(dx) + np.square(dy)
        sep_needed = np.square(neighbors_r + r_i)
        good_candidate = np.all(sq_distances >= sep_needed)
        if good_candidate:
            return xyr_i
    raise RuntimeError('No non-overlapping candidates found. This should not happen.')