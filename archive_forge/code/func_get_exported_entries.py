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
def get_exported_entries(self, category, name=None):
    """
        Return all of the exported entries in a particular category.

        :param category: The category to search for entries.
        :param name: If specified, only entries with that name are returned.
        """
    for dist in self.get_distributions():
        r = dist.exports
        if category in r:
            d = r[category]
            if name is not None:
                if name in d:
                    yield d[name]
            else:
                for v in d.values():
                    yield v