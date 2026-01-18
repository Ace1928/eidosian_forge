import logging
from rasterio._base import get_dataset_driver, driver_can_create, driver_can_create_copy
from rasterio._io import (
from rasterio.windows import WindowMethodsMixin
from rasterio.env import Env, ensure_env
from rasterio.transform import TransformMethodsMixin
from rasterio._path import _UnparsedPath
def get_writer_for_path(path, driver=None):
    """Return the writer class appropriate for the existing dataset."""
    if not driver:
        driver = get_dataset_driver(path)
    return get_writer_for_driver(driver)