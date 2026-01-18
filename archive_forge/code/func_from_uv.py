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
def from_uv(cls, data, kdims=None, vdims=None, **params):
    if kdims is None:
        kdims = ['x', 'y']
    if vdims is None:
        vdims = ['u', 'v']
    dataset = Dataset(data, kdims=kdims, vdims=vdims, **params)
    us, vs = (dataset.dimension_values(i) for i in (2, 3))
    uv_magnitudes = np.hypot(us, vs)
    radians = np.pi / 2 - np.arctan2(-vs, -us)
    repackaged_dataset = {}
    for kdim in kdims:
        repackaged_dataset[kdim] = dataset[kdim]
    repackaged_dataset['Angle'] = radians
    repackaged_dataset['Magnitude'] = uv_magnitudes
    for vdim in vdims[2:]:
        repackaged_dataset[vdim] = dataset[vdim]
    vdims = [Dimension('Angle', cyclic=True, range=(0, 2 * np.pi)), Dimension('Magnitude')] + vdims[2:]
    return cls(repackaged_dataset, kdims=kdims, vdims=vdims, **params)