import html
from urllib.parse import parse_qsl, quote, unquote, urlencode
from paste import request
def js_repr(v):
    if v is None:
        return 'null'
    elif v is False:
        return 'false'
    elif v is True:
        return 'true'
    elif isinstance(v, list):
        return '[%s]' % ', '.join(map(js_repr, v))
    elif isinstance(v, dict):
        return '{%s}' % ', '.join(['%s: %s' % (js_repr(key), js_repr(value)) for key, value in v])
    elif isinstance(v, str):
        return repr(v)
    elif isinstance(v, (float, int)):
        return repr(v)
    elif hasattr(v, '__js_repr__'):
        return v.__js_repr__()
    else:
        raise ValueError("I don't know how to turn %r into a Javascript representation" % v)