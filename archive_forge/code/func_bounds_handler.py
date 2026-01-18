import logging
import os
import re
import click
import rasterio
import rasterio.shutil
from rasterio._path import _parse_path, _UnparsedPath
def bounds_handler(ctx, param, value):
    """Handle different forms of bounds."""
    retval = from_like_context(ctx, param, value)
    if retval is None and value is not None:
        try:
            value = value.strip(', []')
            retval = tuple((float(x) for x in re.split('[,\\s]+', value)))
            assert len(retval) == 4
            return retval
        except Exception:
            raise click.BadParameter('{0!r} is not a valid bounding box representation'.format(value))
    else:
        return retval