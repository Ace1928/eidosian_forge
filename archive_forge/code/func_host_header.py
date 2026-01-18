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
def host_header(self, host, http_request):
    port = http_request.port
    secure = http_request.protocol == 'https'
    if port == 80 and (not secure) or (port == 443 and secure):
        return http_request.host
    return '%s:%s' % (http_request.host, port)