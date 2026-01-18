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
def credentialize(self):
    """Get credentials and configure GDAL

        Note well: this method is a no-op if the GDAL environment
        already has credentials, unless session is not None.

        Returns
        -------
        None

        """
    cred_opts = self.session.get_credential_options()
    self.options.update(**cred_opts)
    setenv(**cred_opts)