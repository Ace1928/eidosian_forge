import param
import numpy as np
from bokeh.models import MercatorTileSource
from cartopy import crs as ccrs
from cartopy.feature import Feature as cFeature
from cartopy.io.img_tiles import GoogleTiles
from cartopy.io.shapereader import Reader
from holoviews.core import Element2D, Dimension, Dataset as HvDataset, NdOverlay, Overlay
from holoviews.core import util
from holoviews.element import (
from holoviews.element.selection import Selection2DExpr
from shapely.geometry.base import BaseGeometry
from shapely.geometry import (
from shapely.ops import unary_union
from ..util import (
@classmethod
def from_xarray(cls, da, crs=None, apply_transform=False, nan_nodata=False, **kwargs):
    """
        Returns an RGB or Image element given an xarray DataArray
        loaded using xr.open_rasterio.

        If a crs attribute is present on the loaded data it will
        attempt to decode it into a cartopy projection otherwise it
        will default to a non-geographic HoloViews element.

        Parameters
        ----------
        da: xarray.DataArray
          DataArray to convert to element
        crs: Cartopy CRS or EPSG string (optional)
          Overrides CRS inferred from the data
        apply_transform: boolean
          Whether to apply affine transform if defined on the data
        nan_nodata: boolean
          If data contains nodata values convert them to NaNs
        **kwargs:
          Keyword arguments passed to the HoloViews/GeoViews element

        Returns
        -------
        element: Image/RGB/QuadMesh element
        """
    return from_xarray(da, crs, apply_transform, **kwargs)