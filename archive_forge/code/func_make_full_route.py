import re
import sys
import six
from six.moves.urllib import parse as urlparse
from routes.util import _url_quote as url_quote, _str_encode, as_unicode
def make_full_route(self):
    """Make a full routelist string for use with non-minimized
        generation"""
    regpath = ''
    for part in self.routelist:
        if isinstance(part, dict):
            regpath += '%(' + part['name'] + ')s'
        else:
            regpath += part
    self.regpath = regpath