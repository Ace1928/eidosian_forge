import re
import warnings
from pprint import pformat
from http.cookies import SimpleCookie
from paste.request import EnvironHeaders, get_cookie_dict, \
from paste.util.multidict import MultiDict, UnicodeMultiDict
from paste.registry import StackedObjectProxy
from paste.response import HeaderDict
from paste.wsgilib import encode_unicode_app_iter
from paste.httpheaders import ACCEPT_LANGUAGE
from paste.util.mimeparse import desired_matches
def charset__del(self):
    try:
        header = self.headers.pop('content-type')
    except KeyError:
        return
    match = _CHARSET_RE.search(header)
    if match:
        header = header[:match.start()] + header[match.end():]
    self.headers['content-type'] = header