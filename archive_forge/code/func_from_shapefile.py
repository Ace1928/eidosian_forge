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
def from_shapefile(cls, shapefile, *args, **kwargs):
    """
        Loads a shapefile from disk and optionally merges
        it with a dataset. See ``from_records`` for full
        signature.

        Parameters
        ----------
        records: list of cartopy.io.shapereader.Record
           Iterator containing Records.
        dataset: holoviews.Dataset
           Any HoloViews Dataset type.
        on: str or list or dict
          A mapping between the attribute names in the records and the
          dimensions in the dataset.
        value: str
          The value dimension in the dataset the values will be drawn
          from.
        index: str or list
          One or more dimensions in the dataset the Shapes will be
          indexed by.
        drop_missing: boolean
          Whether to drop shapes which are missing from the provides
          dataset.

        Returns
        -------
        shapes: Polygons or Path object
          A Polygons or Path object containing the geometries
        """
    reader = Reader(shapefile)
    return cls.from_records(reader.records(), *args, **kwargs)