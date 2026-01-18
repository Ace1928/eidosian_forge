from contextlib import ExitStack
from functools import partial
import math
import numpy as np
import warnings
from affine import Affine
from rasterio.env import env_ctx_if_needed
from rasterio._transform import (
from rasterio.enums import TransformDirection, TransformMethod
from rasterio.control import GroundControlPoint
from rasterio.rpc import RPC
from rasterio.errors import TransformError, RasterioDeprecationWarning
def rowcol(self, xs, ys, zs=None, op=math.floor, precision=None):
    """Get rows and cols coordinates given geographic coordinates.

        Parameters
        ----------
        xs, ys : float or list of float
            Geographic coordinates
        zs : float or list of float, optional
            Height associated with coordinates. Primarily used for RPC based
            coordinate transformations. Ignored for affine based
            transformations. Default: 0.
        op : function, optional (default: math.floor)
            Function to convert fractional pixels to whole numbers (floor,
            ceiling, round)
        precision : int, optional (default: None)
            This parameter is unused, deprecated in rasterio 1.3.0, and
            will be removed in version 2.0.0.

        Raises
        ------
        ValueError
            If input coordinates are not all equal length

        Returns
        -------
        tuple of float or list of float.

        """
    if precision is not None:
        warnings.warn('The precision parameter is unused, deprecated, and will be removed in 2.0.0.', RasterioDeprecationWarning)
    AS_ARR = True if hasattr(xs, '__iter__') else False
    xs, ys, zs = self._ensure_arr_input(xs, ys, zs=zs)
    try:
        new_cols, new_rows = self._transform(xs, ys, zs, transform_direction=TransformDirection.reverse)
        if len(new_rows) == 1 and (not AS_ARR):
            return (op(new_rows[0]), op(new_cols[0]))
        else:
            return ([op(r) for r in new_rows], [op(c) for c in new_cols])
    except TypeError:
        raise TransformError('Invalid inputs')