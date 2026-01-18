import glob
import hashlib
import importlib.util
import itertools
import json
import logging
import os
import pkgutil
import re
import urllib
from urllib import parse as urlparse
from keystoneauth1 import access
from keystoneauth1 import adapter
from keystoneauth1 import discover
from keystoneauth1.identity import base
from oslo_utils import encodeutils
from oslo_utils import importutils
from oslo_utils import strutils
import requests
from cinderclient._i18n import _
from cinderclient import api_versions
from cinderclient import exceptions
import cinderclient.extension
def get_volume_api_from_url(url):
    scheme, netloc, path, query, frag = urlparse.urlsplit(url)
    components = path.split('/')
    for version in _VALID_VERSIONS:
        if version in components:
            return version[1:]
    msg = _("Invalid url: '%(url)s'. It must include one of: %(version)s.") % {'url': url, 'version': ', '.join(_VALID_VERSIONS)}
    raise exceptions.UnsupportedVersion(msg)