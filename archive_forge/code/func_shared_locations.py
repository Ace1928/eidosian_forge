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
@cached_property
def shared_locations(self):
    """
        A dictionary of shared locations whose keys are in the set 'prefix',
        'purelib', 'platlib', 'scripts', 'headers', 'data' and 'namespace'.
        The corresponding value is the absolute path of that category for
        this distribution, and takes into account any paths selected by the
        user at installation time (e.g. via command-line arguments). In the
        case of the 'namespace' key, this would be a list of absolute paths
        for the roots of namespace packages in this distribution.

        The first time this property is accessed, the relevant information is
        read from the SHARED file in the .dist-info directory.
        """
    result = {}
    shared_path = os.path.join(self.path, 'SHARED')
    if os.path.isfile(shared_path):
        with codecs.open(shared_path, 'r', encoding='utf-8') as f:
            lines = f.read().splitlines()
        for line in lines:
            key, value = line.split('=', 1)
            if key == 'namespace':
                result.setdefault(key, []).append(value)
            else:
                result[key] = value
    return result