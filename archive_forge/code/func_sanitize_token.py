from __future__ import absolute_import, division, unicode_literals
import re
import warnings
from xml.sax.saxutils import escape, unescape
from six.moves import urllib_parse as urlparse
from . import base
from ..constants import namespaces, prefixes
def sanitize_token(self, token):
    token_type = token['type']
    if token_type in ('StartTag', 'EndTag', 'EmptyTag'):
        name = token['name']
        namespace = token['namespace']
        if (namespace, name) in self.allowed_elements or (namespace is None and (namespaces['html'], name) in self.allowed_elements):
            return self.allowed_token(token)
        else:
            return self.disallowed_token(token)
    elif token_type == 'Comment':
        pass
    else:
        return token