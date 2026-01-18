from contextlib import ExitStack
import logging
import os
import warnings
from collections import OrderedDict
from fiona import compat, vfs
from fiona.ogrext import Iterator, ItemsIterator, KeysIterator
from fiona.ogrext import Session, WritingSession
from fiona.ogrext import buffer_to_virtual_file, remove_virtual_file, GEOMETRY_TYPES
from fiona.errors import (
from fiona.logutils import FieldSkipLogFilter
from fiona.crs import CRS
from fiona._env import get_gdal_release_name, get_gdal_version_tuple
from fiona.env import env_ctx_if_needed
from fiona.errors import FionaDeprecationWarning
from fiona.drvsupport import (
from fiona.path import Path, vsi_path, parse_path
def guard_driver_mode(self):
    if not self._allow_unsupported_drivers:
        driver = self.session.get_driver()
        if driver not in supported_drivers:
            raise DriverError('unsupported driver: %r' % driver)
        if self.mode not in supported_drivers[driver]:
            raise DriverError('unsupported mode: %r' % self.mode)