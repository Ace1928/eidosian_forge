from __future__ import absolute_import, unicode_literals
import binascii
import hashlib
import hmac
import logging
from oauthlib.common import (extract_params, safe_string_equals, unicode_type,
from . import utils
def normalize_base_string_uri(uri, host=None):
    """**Base String URI**

    Per `section 3.4.1.2`_ of the spec.

    For example, the HTTP request::

        GET /r%20v/X?id=123 HTTP/1.1
        Host: EXAMPLE.COM:80

    is represented by the base string URI: "http://example.com/r%20v/X".

    In another example, the HTTPS request::

        GET /?q=1 HTTP/1.1
        Host: www.example.net:8080

    is represented by the base string URI: "https://www.example.net:8080/".

    .. _`section 3.4.1.2`: https://tools.ietf.org/html/rfc5849#section-3.4.1.2

    The host argument overrides the netloc part of the uri argument.
    """
    if not isinstance(uri, unicode_type):
        raise ValueError('uri must be a unicode object.')
    scheme, netloc, path, params, query, fragment = urlparse.urlparse(uri)
    if not scheme or not netloc:
        raise ValueError('uri must include a scheme and netloc')
    if not path:
        path = '/'
    scheme = scheme.lower()
    netloc = netloc.lower()
    if host is not None:
        netloc = host.lower()
    default_ports = (('http', '80'), ('https', '443'))
    if ':' in netloc:
        host, port = netloc.split(':', 1)
        if (scheme, port) in default_ports:
            netloc = host
    return urlparse.urlunparse((scheme, netloc, path, params, '', ''))