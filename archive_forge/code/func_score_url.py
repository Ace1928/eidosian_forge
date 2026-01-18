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
def score_url(self, url):
    """
        Give an url a score which can be used to choose preferred URLs
        for a given project release.
        """
    t = urlparse(url)
    basename = posixpath.basename(t.path)
    compatible = True
    is_wheel = basename.endswith('.whl')
    is_downloadable = basename.endswith(self.downloadable_extensions)
    if is_wheel:
        compatible = is_compatible(Wheel(basename), self.wheel_tags)
    return (t.scheme == 'https', 'pypi.org' in t.netloc, is_downloadable, is_wheel, compatible, basename)