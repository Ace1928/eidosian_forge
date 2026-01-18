import http.client as httplib
from urllib import parse as urlparse
from urllib.parse import quote
from paste import httpexceptions
from paste.util.converters import aslist
def make_proxy(global_conf, address, allowed_request_methods='', suppress_http_headers=''):
    """
    Make a WSGI application that proxies to another address:

    ``address``
        the full URL ending with a trailing ``/``

    ``allowed_request_methods``:
        a space seperated list of request methods (e.g., ``GET POST``)

    ``suppress_http_headers``
        a space seperated list of http headers (lower case, without
        the leading ``http_``) that should not be passed on to target
        host
    """
    allowed_request_methods = aslist(allowed_request_methods)
    suppress_http_headers = aslist(suppress_http_headers)
    return Proxy(address, allowed_request_methods=allowed_request_methods, suppress_http_headers=suppress_http_headers)