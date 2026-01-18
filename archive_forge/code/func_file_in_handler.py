import logging
import os
import re
import click
import rasterio
import rasterio.shutil
from rasterio._path import _parse_path, _UnparsedPath
def file_in_handler(ctx, param, value):
    """Normalize ordinary filesystem and VFS paths"""
    try:
        path = _parse_path(value)
        if isinstance(path, _UnparsedPath):
            if os.path.exists(path.path) and rasterio.shutil.exists(value):
                return abspath_forward_slashes(path.path)
            else:
                return path.name
        elif path.scheme and path.is_remote:
            return path.name
        elif path.archive:
            if os.path.exists(path.archive) and rasterio.shutil.exists(value):
                archive = abspath_forward_slashes(path.archive)
                return '{}://{}!{}'.format(path.scheme, archive, path.path)
            else:
                raise OSError('Input archive {} does not exist'.format(path.archive))
        elif os.path.exists(path.path) and rasterio.shutil.exists(value):
            return abspath_forward_slashes(path.path)
        else:
            raise OSError('Input file {} does not exist'.format(path.path))
    except Exception:
        raise click.BadParameter('{} is not a valid input file'.format(value))