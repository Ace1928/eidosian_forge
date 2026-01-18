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
def charset__get(self):
    """
        Get/set the charset (in the Content-Type)
        """
    header = self.headers.get('content-type')
    if not header:
        return None
    match = _CHARSET_RE.search(header)
    if match:
        return match.group(1)
    return None