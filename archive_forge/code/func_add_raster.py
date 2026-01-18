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
def add_raster(self, raster_source, **slippy_image_kwargs):
    """
        Add the given raster source to the GeoAxes.

        Parameters
        ----------
        raster_source:
            :class:`cartopy.io.RasterSource` like instance
             ``raster_source`` may be any object which
             implements the RasterSource interface, including
             instances of objects such as
             :class:`~cartopy.io.ogc_clients.WMSRasterSource`
             and
             :class:`~cartopy.io.ogc_clients.WMTSRasterSource`.
             Note that image retrievals are done at draw time,
             not at creation time.

        """
    raster_source.validate_projection(self.projection)
    img = SlippyImageArtist(self, raster_source, **slippy_image_kwargs)
    with self.hold_limits():
        self.add_image(img)
    return img