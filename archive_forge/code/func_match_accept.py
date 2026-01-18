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
def match_accept(self, mimetypes):
    """Return a list of specified mime-types that the browser's HTTP Accept
        header allows in the order provided."""
    return desired_matches(mimetypes, self.environ.get('HTTP_ACCEPT', '*/*'))