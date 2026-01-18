import logging
from rasterio._base import get_dataset_driver, driver_can_create, driver_can_create_copy
from rasterio._io import (
from rasterio.windows import WindowMethodsMixin
from rasterio.env import Env, ensure_env
from rasterio.transform import TransformMethodsMixin
from rasterio._path import _UnparsedPath
def get_writer_for_driver(driver):
    """Return the writer class appropriate for the specified driver."""
    if not driver:
        raise ValueError("'driver' is required to read/write dataset.")
    cls = None
    if driver_can_create(driver):
        cls = DatasetWriter
    elif driver_can_create_copy(driver):
        cls = BufferedDatasetWriter
    return cls