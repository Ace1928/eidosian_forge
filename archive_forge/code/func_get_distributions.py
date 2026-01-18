from __future__ import unicode_literals
import base64
import codecs
import contextlib
import hashlib
import logging
import os
import posixpath
import sys
import zipimport
from . import DistlibException, resources
from .compat import StringIO
from .version import get_scheme, UnsupportedVersionError
from .metadata import (Metadata, METADATA_FILENAME, WHEEL_METADATA_FILENAME,
from .util import (parse_requirement, cached_property, parse_name_and_version,
def get_distributions(self):
    """
        Provides an iterator that looks for distributions and returns
        :class:`InstalledDistribution` or
        :class:`EggInfoDistribution` instances for each one of them.

        :rtype: iterator of :class:`InstalledDistribution` and
                :class:`EggInfoDistribution` instances
        """
    if not self._cache_enabled:
        for dist in self._yield_distributions():
            yield dist
    else:
        self._generate_cache()
        for dist in self._cache.path.values():
            yield dist
        if self._include_egg:
            for dist in self._cache_egg.path.values():
                yield dist