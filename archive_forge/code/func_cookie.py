import base64
import binascii
import ipaddress
import json
import webbrowser
from datetime import datetime
import six
from pymacaroons import Macaroon
from pymacaroons.serializers import json_serializer
import six.moves.http_cookiejar as http_cookiejar
from six.moves.urllib.parse import urlparse
def cookie(url, name, value, expires=None):
    """Return a new Cookie using a slightly more
    friendly API than that provided by six.moves.http_cookiejar

    @param name The cookie name {str}
    @param value The cookie value {str}
    @param url The URL path of the cookie {str}
    @param expires The expiry time of the cookie {datetime}. If provided,
        it must be a naive timestamp in UTC.
    """
    u = urlparse(url)
    domain = u.hostname
    if '.' not in domain and (not _is_ip_addr(domain)):
        domain += '.local'
    port = str(u.port) if u.port is not None else None
    secure = u.scheme == 'https'
    if expires is not None:
        if expires.tzinfo is not None:
            raise ValueError('Cookie expiration must be a naive datetime')
        expires = (expires - datetime(1970, 1, 1)).total_seconds()
    return http_cookiejar.Cookie(version=0, name=name, value=value, port=port, port_specified=port is not None, domain=domain, domain_specified=True, domain_initial_dot=False, path=u.path, path_specified=True, secure=secure, expires=expires, discard=False, comment=None, comment_url=None, rest=None, rfc2109=False)