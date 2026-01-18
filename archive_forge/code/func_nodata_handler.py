import logging
import os
import re
import click
import rasterio
import rasterio.shutil
from rasterio._path import _parse_path, _UnparsedPath
def nodata_handler(ctx, param, value):
    """Return a float or None"""
    if value is None or value.lower() in ['null', 'nil', 'none', 'nada']:
        return None
    else:
        try:
            return float(value)
        except (TypeError, ValueError):
            raise click.BadParameter('{!r} is not a number'.format(value), param=param, param_hint='nodata')