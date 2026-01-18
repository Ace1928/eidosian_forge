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
def crs_handler(ctx, param, value):
    """Get crs value from a template file or command line."""
    retval = options.from_like_context(ctx, param, value)
    if retval is None and value:
        try:
            retval = json.loads(value)
        except ValueError:
            retval = value
        try:
            if isinstance(retval, dict):
                retval = CRS(retval)
            else:
                retval = CRS.from_string(retval)
        except CRSError:
            raise click.BadParameter("'%s' is not a recognized CRS." % retval, param=param, param_hint='crs')
    return retval