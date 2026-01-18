import cgi
from collections.abc import MutableMapping as DictMixin
from urllib import parse as urlparse
from urllib.parse import quote, parse_qsl
from http.cookies import SimpleCookie, CookieError
from paste.util.multidict import MultiDict
def parse_querystring(environ):
    """
    Parses a query string into a list like ``[(name, value)]``.
    Caches this value in case parse_querystring is called again
    for the same request.

    You can pass the result to ``dict()``, but be aware that keys that
    appear multiple times will be lost (only the last value will be
    preserved).

    """
    source = environ.get('QUERY_STRING', '')
    if not source:
        return []
    if 'paste.parsed_querystring' in environ:
        parsed, check_source = environ['paste.parsed_querystring']
        if check_source == source:
            return parsed
    parsed = parse_qsl(source, keep_blank_values=True, strict_parsing=False)
    environ['paste.parsed_querystring'] = (parsed, source)
    return parsed