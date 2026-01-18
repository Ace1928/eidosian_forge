from collections.abc import MutableMapping
import functools
import numpy as np
import matplotlib as mpl
from matplotlib import _api, _docstring
from matplotlib.artist import allow_rasterization
import matplotlib.transforms as mtransforms
import matplotlib.patches as mpatches
import matplotlib.path as mpath
def register_axis(self, axis):
    """
        Register an axis.

        An axis should be registered with its corresponding spine from
        the Axes instance. This allows the spine to clear any axis
        properties when needed.
        """
    self.axis = axis
    self.stale = True