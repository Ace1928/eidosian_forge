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
def canonical_request(self, http_request):
    cr = [http_request.method.upper()]
    cr.append(self.canonical_uri(http_request))
    cr.append(self.canonical_query_string(http_request))
    headers_to_sign = self.headers_to_sign(http_request)
    cr.append(self.canonical_headers(headers_to_sign) + '\n')
    cr.append(self.signed_headers(headers_to_sign))
    cr.append(self.payload(http_request))
    return '\n'.join(cr)