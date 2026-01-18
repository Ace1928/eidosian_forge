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
def provides_distribution(self, name, version=None):
    """
        Iterates over all distributions to find which distributions provide *name*.
        If a *version* is provided, it will be used to filter the results.

        This function only returns the first result found, since no more than
        one values are expected. If the directory is not found, returns ``None``.

        :parameter version: a version specifier that indicates the version
                            required, conforming to the format in ``PEP-345``

        :type name: string
        :type version: string
        """
    matcher = None
    if version is not None:
        try:
            matcher = self._scheme.matcher('%s (%s)' % (name, version))
        except ValueError:
            raise DistlibException('invalid name or version: %r, %r' % (name, version))
    for dist in self.get_distributions():
        if not hasattr(dist, 'provides'):
            logger.debug('No "provides": %s', dist)
        else:
            provided = dist.provides
            for p in provided:
                p_name, p_ver = parse_name_and_version(p)
                if matcher is None:
                    if p_name == name:
                        yield dist
                        break
                elif p_name == name and matcher.match(p_ver):
                    yield dist
                    break