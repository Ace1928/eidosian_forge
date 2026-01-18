import gzip
from io import BytesIO
import json
import logging
import os
import posixpath
import re
import zlib
from . import DistlibException
from .compat import (urljoin, urlparse, urlunparse, url2pathname, pathname2url,
from .database import Distribution, DistributionPath, make_dist
from .metadata import Metadata, MetadataInvalidError
from .util import (cached_property, ensure_slash, split_filename, get_project_data,
from .version import get_scheme, UnsupportedVersionError
from .wheel import Wheel, is_compatible
def prefer_url(self, url1, url2):
    """
        Choose one of two URLs where both are candidates for distribution
        archives for the same version of a distribution (for example,
        .tar.gz vs. zip).

        The current implementation favours https:// URLs over http://, archives
        from PyPI over those from other locations, wheel compatibility (if a
        wheel) and then the archive name.
        """
    result = url2
    if url1:
        s1 = self.score_url(url1)
        s2 = self.score_url(url2)
        if s1 > s2:
            result = url1
        if result != url2:
            logger.debug('Not replacing %r with %r', url1, url2)
        else:
            logger.debug('Replacing %r with %r', url1, url2)
    return result