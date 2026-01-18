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
def list_distinfo_files(self, absolute=False):
    """
        Iterates over the ``installed-files.txt`` entries and returns paths for
        each line if the path is pointing to a file located in the
        ``.egg-info`` directory or one of its subdirectories.

        :parameter absolute: If *absolute* is ``True``, each returned path is
                          transformed into a local absolute path. Otherwise the
                          raw value from ``installed-files.txt`` is returned.
        :type absolute: boolean
        :returns: iterator of paths
        """
    record_path = os.path.join(self.path, 'installed-files.txt')
    if os.path.exists(record_path):
        skip = True
        with codecs.open(record_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line == './':
                    skip = False
                    continue
                if not skip:
                    p = os.path.normpath(os.path.join(self.path, line))
                    if p.startswith(self.path):
                        if absolute:
                            yield p
                        else:
                            yield line