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
def parse_requires_data(data):
    """Create a list of dependencies from a requires.txt file.

            *data*: the contents of a setuptools-produced requires.txt file.
            """
    reqs = []
    lines = data.splitlines()
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.startswith('['):
            logger.warning('Unexpected line: quitting requirement scan: %r', line)
            break
        r = parse_requirement(line)
        if not r:
            logger.warning('Not recognised as a requirement: %r', line)
            continue
        if r.extras:
            logger.warning('extra requirements in requires.txt are not supported')
        if not r.constraints:
            reqs.append(r.name)
        else:
            cons = ', '.join(('%s%s' % c for c in r.constraints))
            reqs.append('%s (%s)' % (r.name, cons))
    return reqs