import os
from rasterio._base import _raster_driver_extensions
from rasterio.env import GDALVersion, ensure_env
def is_blacklisted(name, mode):
    """Returns True if driver `name` and `mode` are blacklisted."""
    return mode in blacklist.get(name, ())