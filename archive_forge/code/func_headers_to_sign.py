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
def headers_to_sign(self, http_request):
    """
        Select the headers from the request that need to be included
        in the StringToSign.
        """
    host_header_value = self.host_header(self.host, http_request)
    headers_to_sign = {'Host': host_header_value}
    for name, value in http_request.headers.items():
        lname = name.lower()
        if lname not in ['authorization']:
            headers_to_sign[name] = value
    return headers_to_sign