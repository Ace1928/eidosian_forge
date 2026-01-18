import cgi
from collections.abc import MutableMapping as DictMixin
from urllib import parse as urlparse
from urllib.parse import quote, parse_qsl
from http.cookies import SimpleCookie, CookieError
from paste.util.multidict import MultiDict
def resolve_relative_url(url, environ):
    """
    Resolve the given relative URL as being relative to the
    location represented by the environment.  This can be used
    for redirecting to a relative path.  Note: if url is already
    absolute, this function will (intentionally) have no effect
    on it.

    """
    cur_url = construct_url(environ, with_query_string=False)
    return urlparse.urljoin(cur_url, url)