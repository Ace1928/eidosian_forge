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
def safename(filename):
    """Return a filename suitable for the cache.
    Strips dangerous and common characters to create a filename we
    can use to store the cache in.
    """
    if isinstance(filename, bytes):
        filename_bytes = filename
        filename = filename.decode('utf-8')
    else:
        filename_bytes = filename.encode('utf-8')
    filemd5 = _md5(filename_bytes).hexdigest()
    filename = re_url_scheme.sub('', filename)
    filename = re_unsafe.sub('', filename)
    filename = filename[:90]
    return ','.join((filename, filemd5))