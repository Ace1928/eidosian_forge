import html
from urllib.parse import parse_qsl, quote, unquote, urlencode
from paste import request
def addpath(self, *paths):
    u = self
    for path in paths:
        path = str(path).lstrip('/')
        new_url = u.url
        if not new_url.endswith('/'):
            new_url += '/'
        u = u.__class__(new_url + path, vars=u.vars, attrs=u.attrs, params=u.original_params)
    return u