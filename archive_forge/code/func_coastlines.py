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
def coastlines(self, resolution='auto', color='black', **kwargs):
    """
        Add coastal **outlines** to the current axes from the Natural Earth
        "coastline" shapefile collection.

        Parameters
        ----------
        resolution : str or :class:`cartopy.feature.Scaler`, optional
            A named resolution to use from the Natural Earth
            dataset. Currently can be one of "auto" (default), "110m", "50m",
            and "10m", or a Scaler object.  If "auto" is selected, the
            resolution is defined by `~cartopy.feature.auto_scaler`.

        """
    kwargs['edgecolor'] = color
    kwargs['facecolor'] = 'none'
    feature = cartopy.feature.COASTLINE
    if resolution != 'auto':
        feature = feature.with_scale(resolution)
    return self.add_feature(feature, **kwargs)