from collections.abc import MutableMapping
import functools
import numpy as np
import matplotlib as mpl
from matplotlib import _api, _docstring
from matplotlib.artist import allow_rasterization
import matplotlib.transforms as mtransforms
import matplotlib.patches as mpatches
import matplotlib.path as mpath
def set_patch_circle(self, center, radius):
    """Set the spine to be circular."""
    self._patch_type = 'circle'
    self._center = center
    self._width = radius * 2
    self._height = radius * 2
    self.set_transform(self.axes.transAxes)
    self.stale = True