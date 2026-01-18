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
def list_url_unsafe_chars(name):
    """Return a list of the reserved characters."""
    reserved_chars = ''
    for i in name:
        if i in URL_RESERVED_CHARS:
            reserved_chars += i
    return reserved_chars