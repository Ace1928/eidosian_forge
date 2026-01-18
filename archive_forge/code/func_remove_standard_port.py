import base64
import collections.abc
import contextlib
import grp
import hashlib
import itertools
import os
import pwd
import uuid
from cryptography import x509
from oslo_log import log
from oslo_serialization import jsonutils
from oslo_utils import reflection
from oslo_utils import strutils
from oslo_utils import timeutils
import urllib
from keystone.common import password_hashing
import keystone.conf
from keystone import exception
from keystone.i18n import _
def remove_standard_port(url):
    o = urllib.parse.urlparse(url)
    separator = ':'
    host, separator, port = o.netloc.partition(separator)
    if o.scheme.lower() == 'http' and port == '80':
        o = o._replace(netloc=host)
    if o.scheme.lower() == 'https' and port == '443':
        o = o._replace(netloc=host)
    return urllib.parse.urlunparse(o)