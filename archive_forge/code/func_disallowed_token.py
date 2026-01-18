from __future__ import absolute_import, division, unicode_literals
import re
import warnings
from xml.sax.saxutils import escape, unescape
from six.moves import urllib_parse as urlparse
from . import base
from ..constants import namespaces, prefixes
def disallowed_token(self, token):
    token_type = token['type']
    if token_type == 'EndTag':
        token['data'] = '</%s>' % token['name']
    elif token['data']:
        assert token_type in ('StartTag', 'EmptyTag')
        attrs = []
        for (ns, name), v in token['data'].items():
            attrs.append(' %s="%s"' % (name if ns is None else '%s:%s' % (prefixes[ns], name), escape(v)))
        token['data'] = '<%s%s>' % (token['name'], ''.join(attrs))
    else:
        token['data'] = '<%s>' % token['name']
    if token.get('selfClosing'):
        token['data'] = token['data'][:-1] + '/>'
    token['type'] = 'Characters'
    del token['name']
    return token