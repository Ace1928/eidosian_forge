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
@property
def crs_wkt(self):
    """Returns a WKT string."""
    if self._crs_wkt is None and self.session:
        self._crs_wkt = self.session.get_crs_wkt()
    return self._crs_wkt