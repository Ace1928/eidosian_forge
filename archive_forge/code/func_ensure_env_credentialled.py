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
def ensure_env_credentialled(f):
    """DEPRECATED alias for ensure_env_with_credentials"""
    warnings.warn('Please use ensure_env_with_credentials instead', RasterioDeprecationWarning)
    return ensure_env_with_credentials(f)