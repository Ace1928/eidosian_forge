import cgi
from collections.abc import MutableMapping as DictMixin
from urllib import parse as urlparse
from urllib.parse import quote, parse_qsl
from http.cookies import SimpleCookie, CookieError
from paste.util.multidict import MultiDict
def path_info_split(path_info):
    """
    Splits off the first segment of the path.  Returns (first_part,
    rest_of_path).  first_part can be None (if PATH_INFO is empty), ''
    (if PATH_INFO is '/'), or a name without any /'s.  rest_of_path
    can be '' or a string starting with /.

    """
    if not path_info:
        return (None, '')
    assert path_info.startswith('/'), 'PATH_INFO should start with /: %r' % path_info
    path_info = path_info.lstrip('/')
    if '/' in path_info:
        first, rest = path_info.split('/', 1)
        return (first, '/' + rest)
    else:
        return (path_info, '')