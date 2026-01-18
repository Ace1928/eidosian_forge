from collections.abc import MutableMapping
import functools
import numpy as np
import matplotlib as mpl
from matplotlib import _api, _docstring
from matplotlib.artist import allow_rasterization
import matplotlib.transforms as mtransforms
import matplotlib.patches as mpatches
import matplotlib.path as mpath
@classmethod
def circular_spine(cls, axes, center, radius, **kwargs):
    """Create and return a circular `Spine`."""
    path = mpath.Path.unit_circle()
    spine_type = 'circle'
    result = cls(axes, spine_type, path, **kwargs)
    result.set_patch_circle(center, radius)
    return result