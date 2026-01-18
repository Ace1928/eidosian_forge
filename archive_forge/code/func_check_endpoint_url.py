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
def check_endpoint_url(url):
    """Check substitution of url.

    The invalid urls are as follows:
    urls with substitutions that is not in the whitelist

    Check the substitutions in the URL to make sure they are valid
    and on the whitelist.

    :param str url: the URL to validate
    :rtype: None
    :raises keystone.exception.URLValidationError: if the URL is invalid
    """
    substitutions = dict(zip(WHITELISTED_PROPERTIES, itertools.repeat('')))
    try:
        url.replace('$(', '%(') % substitutions
    except (KeyError, TypeError, ValueError):
        raise exception.URLValidationError(url=url)