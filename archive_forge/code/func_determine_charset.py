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
def determine_charset(self):
    """
        Determine the encoding as specified by the Content-Type's charset
        parameter, if one is set
        """
    charset_match = _CHARSET_RE.search(self.headers.get('Content-Type', ''))
    if charset_match:
        return charset_match.group(1)