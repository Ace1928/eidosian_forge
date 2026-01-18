from collections.abc import Mapping
import functools
import numpy as np
from numpy import ma
import matplotlib as mpl
from matplotlib import _api, colors, cbook, scale
from matplotlib._cm import datad
from matplotlib._cm_listed import cmaps as cmaps_listed
def get_clim(self):
    """
        Return the values (min, max) that are mapped to the colormap limits.
        """
    return (self.norm.vmin, self.norm.vmax)