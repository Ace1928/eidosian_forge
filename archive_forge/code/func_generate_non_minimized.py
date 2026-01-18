import re
import sys
import six
from six.moves.urllib import parse as urlparse
from routes.util import _url_quote as url_quote, _str_encode, as_unicode
def generate_non_minimized(self, kargs):
    """Generate a non-minimal version of the URL"""
    for k in self.maxkeys - self.minkeys:
        if k not in kargs:
            return False
        elif self.make_unicode(kargs[k]) != self.make_unicode(self.defaults[k]):
            return False
    for arg in self.minkeys:
        if arg not in kargs or kargs[arg] is None:
            if arg in self.dotkeys:
                kargs[arg] = ''
            else:
                return False
    for k in kargs:
        if k in self.maxkeys:
            if k in self.dotkeys:
                if kargs[k]:
                    kargs[k] = url_quote('.' + as_unicode(kargs[k], self.encoding), self.encoding)
            else:
                kargs[k] = url_quote(as_unicode(kargs[k], self.encoding), self.encoding)
    return self.regpath % kargs