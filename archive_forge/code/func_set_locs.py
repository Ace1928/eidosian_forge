import matplotlib as mpl
from matplotlib.ticker import Formatter, MaxNLocator
import numpy as np
import cartopy.crs as ccrs
from cartopy.mpl.geoaxes import GeoAxes
def set_locs(self, locs):
    _PlateCarreeFormatter.set_locs(self, self._fix_lons(locs))