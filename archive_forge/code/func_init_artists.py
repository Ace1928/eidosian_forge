import copy
import numpy as np
import param
import matplotlib.ticker as mticker
from cartopy import crs as ccrs
from cartopy.io.img_tiles import GoogleTiles, QuadtreeTiles
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from holoviews.core import Store, HoloMap, Layout, Overlay, Element, NdLayout
from holoviews.core import util
from holoviews.core.data import GridInterface
from holoviews.core.options import SkipRendering, Options
from holoviews.plotting.mpl import (
from holoviews.plotting.mpl.util import get_raster_array, wrap_formatter
from ...element import (
from ...util import geo_mesh, poly_types
from ..plot import ProjectionPlot
from ...operation import (
from .chart import WindBarbsPlot
def init_artists(self, ax, plot_args, plot_kwargs):
    if isinstance(plot_args[0], GoogleTiles):
        if 'artist' in self.handles:
            return {'artist': self.handles['artist']}
        img = ax.add_image(*plot_args, **plot_kwargs)
        return {'artist': img or plot_args[0]}
    return {'artist': ax.add_wmts(*plot_args, **plot_kwargs)}