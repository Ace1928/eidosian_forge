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
def get_distinfo_file(self, path):
    """
        Returns a path located under the ``.dist-info`` directory. Returns a
        string representing the path.

        :parameter path: a ``'/'``-separated path relative to the
                         ``.dist-info`` directory or an absolute path;
                         If *path* is an absolute path and doesn't start
                         with the ``.dist-info`` directory path,
                         a :class:`DistlibException` is raised
        :type path: str
        :rtype: str
        """
    if path.find(os.sep) >= 0:
        distinfo_dirname, path = path.split(os.sep)[-2:]
        if distinfo_dirname != self.path.split(os.sep)[-1]:
            raise DistlibException('dist-info file %r does not belong to the %r %s distribution' % (path, self.name, self.version))
    if path not in DIST_FILES:
        raise DistlibException('invalid path for a dist-info file: %r at %r' % (path, self.path))
    return os.path.join(self.path, path)