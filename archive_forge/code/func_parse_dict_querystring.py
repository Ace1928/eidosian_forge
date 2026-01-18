import cgi
from collections.abc import MutableMapping as DictMixin
from urllib import parse as urlparse
from urllib.parse import quote, parse_qsl
from http.cookies import SimpleCookie, CookieError
from paste.util.multidict import MultiDict
def parse_dict_querystring(environ):
    """Parses a query string like parse_querystring, but returns a MultiDict

    Caches this value in case parse_dict_querystring is called again
    for the same request.

    Example::

        >>> environ = {'QUERY_STRING': 'day=Monday&user=fred&user=jane'}
        >>> parsed = parse_dict_querystring(environ)

        >>> parsed['day']
        'Monday'
        >>> parsed['user']
        'fred'
        >>> parsed.getall('user')
        ['fred', 'jane']

    """
    source = environ.get('QUERY_STRING', '')
    if not source:
        return MultiDict()
    if 'paste.parsed_dict_querystring' in environ:
        parsed, check_source = environ['paste.parsed_dict_querystring']
        if check_source == source:
            return parsed
    parsed = parse_qsl(source, keep_blank_values=True, strict_parsing=False)
    multi = MultiDict(parsed)
    environ['paste.parsed_dict_querystring'] = (multi, source)
    return multi