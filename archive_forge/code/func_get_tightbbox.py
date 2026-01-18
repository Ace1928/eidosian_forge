from collections import defaultdict
import functools
import itertools
import math
import textwrap
import numpy as np
import matplotlib as mpl
from matplotlib import _api, cbook, _docstring, _preprocess_data
import matplotlib.artist as martist
import matplotlib.axes as maxes
import matplotlib.collections as mcoll
import matplotlib.colors as mcolors
import matplotlib.image as mimage
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.container as mcontainer
import matplotlib.transforms as mtransforms
from matplotlib.axes import Axes
from matplotlib.axes._base import _axis_method_wrapper, _process_plot_format
from matplotlib.transforms import Bbox
from matplotlib.tri._triangulation import Triangulation
from . import art3d
from . import proj3d
from . import axis3d
@_api.make_keyword_only('3.8', 'call_axes_locator')
def get_tightbbox(self, renderer=None, call_axes_locator=True, bbox_extra_artists=None, *, for_layout_only=False):
    ret = super().get_tightbbox(renderer, call_axes_locator=call_axes_locator, bbox_extra_artists=bbox_extra_artists, for_layout_only=for_layout_only)
    batch = [ret]
    if self._axis3don:
        for axis in self._axis_map.values():
            if axis.get_visible():
                axis_bb = martist._get_tightbbox_for_layout_only(axis, renderer)
                if axis_bb:
                    batch.append(axis_bb)
    return mtransforms.Bbox.union(batch)