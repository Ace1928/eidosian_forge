import itertools
import logging
import os
import posixpath
import urllib.parse
from typing import List
from pip._vendor.packaging.utils import canonicalize_name
from pip._internal.models.index import PyPI
from pip._internal.utils.compat import has_tls
from pip._internal.utils.misc import normalize_path, redact_auth_from_url
def mkurl_pypi_url(url: str) -> str:
    loc = posixpath.join(url, urllib.parse.quote(canonicalize_name(project_name)))
    if not loc.endswith('/'):
        loc = loc + '/'
    return loc