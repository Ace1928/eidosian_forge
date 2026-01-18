import logging
import os
import re
import click
import rasterio
import rasterio.shutil
from rasterio._path import _parse_path, _UnparsedPath
def files_inout_handler(ctx, param, value):
    """Process and validate input file names"""
    return tuple((file_in_handler(ctx, param, item) for item in value[:-1])) + tuple(value[-1:])