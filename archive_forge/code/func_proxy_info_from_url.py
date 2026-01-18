import base64
import calendar
import copy
import email
import email.feedparser
from email import header
import email.message
import email.utils
import errno
from gettext import gettext as _
import gzip
from hashlib import md5 as _md5
from hashlib import sha1 as _sha
import hmac
import http.client
import io
import os
import random
import re
import socket
import ssl
import sys
import time
import urllib.parse
import zlib
from . import auth
from .error import *
from .iri2uri import iri2uri
from httplib2 import certs
def proxy_info_from_url(url, method='http', noproxy=None):
    """Construct a ProxyInfo from a URL (such as http_proxy env var)
    """
    url = urllib.parse.urlparse(url)
    proxy_type = 3
    pi = ProxyInfo(proxy_type=proxy_type, proxy_host=url.hostname, proxy_port=url.port or dict(https=443, http=80)[method], proxy_user=url.username or None, proxy_pass=url.password or None, proxy_headers=None)
    bypass_hosts = []
    if noproxy is None:
        noproxy = os.environ.get('no_proxy', os.environ.get('NO_PROXY', ''))
    if noproxy == '*':
        bypass_hosts = AllHosts
    elif noproxy.strip():
        bypass_hosts = noproxy.split(',')
        bypass_hosts = tuple(filter(bool, bypass_hosts))
    pi.bypass_hosts = bypass_hosts
    return pi