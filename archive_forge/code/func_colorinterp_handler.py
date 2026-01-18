from collections import OrderedDict
import json
import warnings
import click
import rasterio
import rasterio.crs
from rasterio.crs import CRS
from rasterio.dtypes import in_dtype_range
from rasterio.enums import ColorInterp
from rasterio.errors import CRSError
from rasterio.rio import options
from rasterio.transform import guard_transform
def colorinterp_handler(ctx, param, value):
    """Validate a string like ``red,green,blue,alpha`` and convert to
    a tuple.  Also handle ``RGB`` and ``RGBA``.
    """
    if value is None:
        return value
    elif value.lower() == 'like':
        return options.from_like_context(ctx, param, value)
    elif value.lower() == 'rgb':
        return (ColorInterp.red, ColorInterp.green, ColorInterp.blue)
    elif value.lower() == 'rgba':
        return (ColorInterp.red, ColorInterp.green, ColorInterp.blue, ColorInterp.alpha)
    else:
        colorinterp = tuple(value.split(','))
        for ci in colorinterp:
            if ci not in ColorInterp.__members__:
                raise click.BadParameter("color interpretation '{ci}' is invalid.  Must be one of: {valid}".format(ci=ci, valid=', '.join(ColorInterp.__members__)))
        return tuple((ColorInterp[ci] for ci in colorinterp))