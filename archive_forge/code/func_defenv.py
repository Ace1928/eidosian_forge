from contextlib import ExitStack
from functools import wraps, total_ordering
from inspect import getfullargspec as getargspec
import logging
import os
import re
import threading
import warnings
import attr
from rasterio._env import (
from rasterio._version import gdal_version
from rasterio.errors import EnvError, GDALVersionError, RasterioDeprecationWarning
from rasterio.session import Session, DummySession
def defenv(**options):
    """Create a default environment if necessary."""
    if local._env:
        log.debug('GDAL environment exists: %r', local._env)
    else:
        log.debug('No GDAL environment exists')
        local._env = GDALEnv()
        local._env.update_config_options(**options)
        log.debug('New GDAL environment %r created', local._env)
    local._env.start()