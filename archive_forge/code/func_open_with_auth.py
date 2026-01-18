import sys
import os
import re
import io
import shutil
import socket
import base64
import hashlib
import itertools
import configparser
import html
import http.client
import urllib.parse
import urllib.request
import urllib.error
from functools import wraps
import setuptools
from pkg_resources import (
from distutils import log
from distutils.errors import DistutilsError
from fnmatch import translate
from setuptools.wheel import Wheel
from setuptools.extern.more_itertools import unique_everseen
def open_with_auth(url, opener=urllib.request.urlopen):
    """Open a urllib2 request, handling HTTP authentication"""
    parsed = urllib.parse.urlparse(url)
    scheme, netloc, path, params, query, frag = parsed
    if netloc.endswith(':'):
        raise http.client.InvalidURL("nonnumeric port: ''")
    if scheme in ('http', 'https'):
        auth, address = _splituser(netloc)
    else:
        auth = None
    if not auth:
        cred = PyPIConfig().find_credential(url)
        if cred:
            auth = str(cred)
            info = (cred.username, url)
            log.info('Authenticating as %s for %s (from .pypirc)', *info)
    if auth:
        auth = 'Basic ' + _encode_auth(auth)
        parts = (scheme, address, path, params, query, frag)
        new_url = urllib.parse.urlunparse(parts)
        request = urllib.request.Request(new_url)
        request.add_header('Authorization', auth)
    else:
        request = urllib.request.Request(url)
    request.add_header('User-Agent', user_agent)
    fp = opener(request)
    if auth:
        s2, h2, path2, param2, query2, frag2 = urllib.parse.urlparse(fp.url)
        if s2 == scheme and h2 == address:
            parts = (s2, netloc, path2, param2, query2, frag2)
            fp.url = urllib.parse.urlunparse(parts)
    return fp