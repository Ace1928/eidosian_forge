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
def parse_requires_path(req_path):
    """Create a list of dependencies from a requires.txt file.

            *req_path*: the path to a setuptools-produced requires.txt file.
            """
    reqs = []
    try:
        with codecs.open(req_path, 'r', 'utf-8') as fp:
            reqs = parse_requires_data(fp.read())
    except IOError:
        pass
    return reqs