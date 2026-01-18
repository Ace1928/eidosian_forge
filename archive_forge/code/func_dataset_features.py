import logging
import math
import os
import warnings
import numpy as np
import rasterio
from rasterio import warp
from rasterio._base import DatasetBase
from rasterio._features import _shapes, _sieve, _rasterize, _bounds
from rasterio.dtypes import validate_dtype, can_cast_dtype, get_minimum_dtype, _getnpdtype
from rasterio.enums import MergeAlg
from rasterio.env import ensure_env, GDALVersion
from rasterio.errors import ShapeSkipWarning
from rasterio.rio.helpers import coords
from rasterio.transform import Affine
from rasterio.transform import IDENTITY, guard_transform
from rasterio.windows import Window
def dataset_features(src, bidx=None, sampling=1, band=True, as_mask=False, with_nodata=False, geographic=True, precision=-1):
    """Yield GeoJSON features for the dataset

    The geometries are polygons bounding contiguous regions of the same raster value.

    Parameters
    ----------
    src: Rasterio Dataset

    bidx: int
        band index

    sampling: int (DEFAULT: 1)
        Inverse of the sampling fraction; a value of 10 decimates

    band: boolean (DEFAULT: True)
        extract features from a band (True) or a mask (False)

    as_mask: boolean (DEFAULT: False)
        Interpret band as a mask and output only one class of valid data shapes?

    with_nodata: boolean (DEFAULT: False)
        Include nodata regions?

    geographic: str (DEFAULT: True)
        Output shapes in EPSG:4326? Otherwise use the native CRS.

    precision: int (DEFAULT: -1)
        Decimal precision of coordinates. -1 for full float precision output

    Yields
    ------
    GeoJSON-like Feature dictionaries for shapes found in the given band
    """
    if bidx is not None and bidx > src.count:
        raise ValueError('bidx is out of range for raster')
    img = None
    msk = None
    transform = src.transform
    if sampling > 1:
        shape = (int(math.ceil(src.height / sampling)), int(math.ceil(src.width / sampling)))
        x_sampling = src.width / shape[1]
        y_sampling = src.height / shape[0]
        transform *= Affine.translation(src.width % x_sampling, src.height % y_sampling)
        transform *= Affine.scale(x_sampling, y_sampling)
    if not band or (band and (not as_mask) and (not with_nodata)):
        if sampling == 1:
            msk = src.read_masks(bidx)
        else:
            msk_shape = shape
            if bidx is None:
                msk = np.zeros((src.count,) + msk_shape, 'uint8')
            else:
                msk = np.zeros(msk_shape, 'uint8')
            msk = src.read_masks(bidx, msk)
        if bidx is None:
            msk = np.logical_or.reduce(msk).astype('uint8')
        img = msk
    if band:
        if sampling == 1:
            img = src.read(bidx, masked=False)
        else:
            img = np.zeros(shape, dtype=src.dtypes[src.indexes.index(bidx)])
            img = src.read(bidx, img, masked=False)
    if as_mask:
        tmp = np.ones_like(img, 'uint8') * 255
        tmp[img == 0] = 0
        img = tmp
        if not with_nodata:
            msk = tmp
    kwargs = {'transform': transform}
    if not with_nodata:
        kwargs['mask'] = msk
    src_basename = os.path.basename(src.name)
    for i, (g, val) in enumerate(rasterio.features.shapes(img, **kwargs)):
        if geographic:
            g = warp.transform_geom(src.crs, 'EPSG:4326', g, antimeridian_cutting=True, precision=precision)
        xs, ys = zip(*coords(g))
        yield {'type': 'Feature', 'id': '{0}:{1}'.format(src_basename, i), 'properties': {'val': val, 'filename': src_basename}, 'bbox': [min(xs), min(ys), max(xs), max(ys)], 'geometry': g}