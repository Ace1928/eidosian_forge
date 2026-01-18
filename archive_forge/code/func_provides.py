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
@property
def provides(self):
    """
        A set of distribution names and versions provided by this distribution.
        :return: A set of "name (version)" strings.
        """
    plist = self.metadata.provides
    s = '%s (%s)' % (self.name, self.version)
    if s not in plist:
        plist.append(s)
    return plist