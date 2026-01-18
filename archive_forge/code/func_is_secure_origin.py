import email.utils
import io
import ipaddress
import json
import logging
import mimetypes
import os
import platform
import shutil
import subprocess
import sys
import urllib.parse
import warnings
from typing import (
from pip._vendor import requests, urllib3
from pip._vendor.cachecontrol import CacheControlAdapter as _BaseCacheControlAdapter
from pip._vendor.requests.adapters import DEFAULT_POOLBLOCK, BaseAdapter
from pip._vendor.requests.adapters import HTTPAdapter as _BaseHTTPAdapter
from pip._vendor.requests.models import PreparedRequest, Response
from pip._vendor.requests.structures import CaseInsensitiveDict
from pip._vendor.urllib3.connectionpool import ConnectionPool
from pip._vendor.urllib3.exceptions import InsecureRequestWarning
from pip import __version__
from pip._internal.metadata import get_default_environment
from pip._internal.models.link import Link
from pip._internal.network.auth import MultiDomainBasicAuth
from pip._internal.network.cache import SafeFileCache
from pip._internal.utils.compat import has_tls
from pip._internal.utils.glibc import libc_ver
from pip._internal.utils.misc import build_url_from_netloc, parse_netloc
from pip._internal.utils.urls import url_to_path
def is_secure_origin(self, location: Link) -> bool:
    parsed = urllib.parse.urlparse(str(location))
    origin_protocol, origin_host, origin_port = (parsed.scheme, parsed.hostname, parsed.port)
    origin_protocol = origin_protocol.rsplit('+', 1)[-1]
    for secure_origin in self.iter_secure_origins():
        secure_protocol, secure_host, secure_port = secure_origin
        if origin_protocol != secure_protocol and secure_protocol != '*':
            continue
        try:
            addr = ipaddress.ip_address(origin_host or '')
            network = ipaddress.ip_network(secure_host)
        except ValueError:
            if origin_host and origin_host.lower() != secure_host.lower() and (secure_host != '*'):
                continue
        else:
            if addr not in network:
                continue
        if origin_port != secure_port and secure_port != '*' and (secure_port is not None):
            continue
        return True
    logger.warning("The repository located at %s is not a trusted or secure host and is being ignored. If this repository is available via HTTPS we recommend you use HTTPS instead, otherwise you may silence this warning and allow it anyway with '--trusted-host %s'.", origin_host, origin_host)
    return False