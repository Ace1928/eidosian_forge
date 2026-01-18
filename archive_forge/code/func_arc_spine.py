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
def arc_spine(cls, axes, spine_type, center, radius, theta1, theta2, **kwargs):
    """Create and return an arc `Spine`."""
    path = mpath.Path.arc(theta1, theta2)
    result = cls(axes, spine_type, path, **kwargs)
    result.set_patch_arc(center, radius, theta1, theta2)
    return result