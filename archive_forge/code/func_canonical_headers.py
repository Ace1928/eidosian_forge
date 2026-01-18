import base64
import boto
import boto.auth_handler
import boto.exception
import boto.plugin
import boto.utils
import copy
import datetime
from email.utils import formatdate
import hmac
import os
import posixpath
from boto.compat import urllib, encodebytes, parse_qs_safe, urlparse, six
from boto.auth_handler import AuthHandler
from boto.exception import BotoClientError
from boto.utils import get_utf8able_str
def canonical_headers(self, headers_to_sign):
    """
        Return the headers that need to be included in the StringToSign
        in their canonical form by converting all header keys to lower
        case, sorting them in alphabetical order and then joining
        them into a string, separated by newlines.
        """
    canonical = []
    normalized_headers = {}
    for header in headers_to_sign:
        c_name = header.lower().strip()
        raw_value = str(headers_to_sign[header])
        if '"' in raw_value:
            c_value = raw_value.strip()
        else:
            c_value = ' '.join(raw_value.strip().split())
        normalized_headers[c_name] = c_value
    for key in sorted(normalized_headers):
        canonical.append('%s:%s' % (key, normalized_headers[key]))
    return '\n'.join(canonical)