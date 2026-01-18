import copy
import time
import calendar
from ._internal_utils import to_native_string
from .compat import cookielib, urlparse, urlunparse, Morsel, MutableMapping
def morsel_to_cookie(morsel):
    """Convert a Morsel object into a Cookie containing the one k/v pair."""
    expires = None
    if morsel['max-age']:
        try:
            expires = int(time.time() + int(morsel['max-age']))
        except ValueError:
            raise TypeError('max-age: %s must be integer' % morsel['max-age'])
    elif morsel['expires']:
        time_template = '%a, %d-%b-%Y %H:%M:%S GMT'
        expires = calendar.timegm(time.strptime(morsel['expires'], time_template))
    return create_cookie(comment=morsel['comment'], comment_url=bool(morsel['comment']), discard=False, domain=morsel['domain'], expires=expires, name=morsel.key, path=morsel['path'], port=None, rest={'HttpOnly': morsel['httponly']}, rfc2109=False, secure=bool(morsel['secure']), value=morsel.value, version=morsel['version'] or 0)