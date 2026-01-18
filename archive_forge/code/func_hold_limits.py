import collections
import contextlib
import functools
import json
import os
from pathlib import Path
import warnings
import weakref
import matplotlib as mpl
import matplotlib.artist
import matplotlib.axes
import matplotlib.contour
from matplotlib.image import imread
import matplotlib.patches as mpatches
import matplotlib.path as mpath
import matplotlib.spines as mspines
import matplotlib.transforms as mtransforms
import numpy as np
import numpy.ma as ma
import shapely.geometry as sgeom
from cartopy import config
import cartopy.crs as ccrs
import cartopy.feature
from cartopy.mpl import _MPL_38
import cartopy.mpl.contour
import cartopy.mpl.feature_artist as feature_artist
import cartopy.mpl.geocollection
import cartopy.mpl.patch as cpatch
from cartopy.mpl.slippy_image_artist import SlippyImageArtist
@contextlib.contextmanager
def hold_limits(self, hold=True):
    """
        Keep track of the original view and data limits for the life of this
        context manager, optionally reverting any changes back to the original
        values after the manager exits.

        Parameters
        ----------
        hold: bool, optional
            Whether to revert the data and view limits after the
            context manager exits.  Defaults to True.

        """
    with contextlib.ExitStack() as stack:
        if hold:
            stack.callback(self.dataLim.set_points, self.dataLim.frozen().get_points())
            stack.callback(self.viewLim.set_points, self.viewLim.frozen().get_points())
            stack.callback(setattr, self, 'ignore_existing_data_limits', self.ignore_existing_data_limits)
            stack.callback(self.set_autoscalex_on, self.get_autoscalex_on())
            stack.callback(self.set_autoscaley_on, self.get_autoscaley_on())
        yield