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
def convert_url_to_download_info(self, url, project_name):
    """
        See if a URL is a candidate for a download URL for a project (the URL
        has typically been scraped from an HTML page).

        If it is, a dictionary is returned with keys "name", "version",
        "filename" and "url"; otherwise, None is returned.
        """

    def same_project(name1, name2):
        return normalize_name(name1) == normalize_name(name2)
    result = None
    scheme, netloc, path, params, query, frag = urlparse(url)
    if frag.lower().startswith('egg='):
        logger.debug('%s: version hint in fragment: %r', project_name, frag)
    m = HASHER_HASH.match(frag)
    if m:
        algo, digest = m.groups()
    else:
        algo, digest = (None, None)
    origpath = path
    if path and path[-1] == '/':
        path = path[:-1]
    if path.endswith('.whl'):
        try:
            wheel = Wheel(path)
            if not is_compatible(wheel, self.wheel_tags):
                logger.debug('Wheel not compatible: %s', path)
            else:
                if project_name is None:
                    include = True
                else:
                    include = same_project(wheel.name, project_name)
                if include:
                    result = {'name': wheel.name, 'version': wheel.version, 'filename': wheel.filename, 'url': urlunparse((scheme, netloc, origpath, params, query, '')), 'python-version': ', '.join(['.'.join(list(v[2:])) for v in wheel.pyver])}
        except Exception:
            logger.warning('invalid path for wheel: %s', path)
    elif not path.endswith(self.downloadable_extensions):
        logger.debug('Not downloadable: %s', path)
    else:
        path = filename = posixpath.basename(path)
        for ext in self.downloadable_extensions:
            if path.endswith(ext):
                path = path[:-len(ext)]
                t = self.split_filename(path, project_name)
                if not t:
                    logger.debug('No match for project/version: %s', path)
                else:
                    name, version, pyver = t
                    if not project_name or same_project(project_name, name):
                        result = {'name': name, 'version': version, 'filename': filename, 'url': urlunparse((scheme, netloc, origpath, params, query, ''))}
                        if pyver:
                            result['python-version'] = pyver
                break
    if result and algo:
        result['%s_digest' % algo] = digest
    return result