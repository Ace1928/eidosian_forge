from __future__ import absolute_import, division, unicode_literals
import re
import warnings
from xml.sax.saxutils import escape, unescape
from six.moves import urllib_parse as urlparse
from . import base
from ..constants import namespaces, prefixes
def allowed_token(self, token):
    if 'data' in token:
        attrs = token['data']
        attr_names = set(attrs.keys())
        for to_remove in attr_names - self.allowed_attributes:
            del token['data'][to_remove]
            attr_names.remove(to_remove)
        for attr in attr_names & self.attr_val_is_uri:
            assert attr in attrs
            val_unescaped = re.sub('[`\x00- \x7f-\xa0\\s]+', '', unescape(attrs[attr])).lower()
            val_unescaped = val_unescaped.replace('�', '')
            try:
                uri = urlparse.urlparse(val_unescaped)
            except ValueError:
                uri = None
                del attrs[attr]
            if uri and uri.scheme:
                if uri.scheme not in self.allowed_protocols:
                    del attrs[attr]
                if uri.scheme == 'data':
                    m = data_content_type.match(uri.path)
                    if not m:
                        del attrs[attr]
                    elif m.group('content_type') not in self.allowed_content_types:
                        del attrs[attr]
        for attr in self.svg_attr_val_allows_ref:
            if attr in attrs:
                attrs[attr] = re.sub('url\\s*\\(\\s*[^#\\s][^)]+?\\)', ' ', unescape(attrs[attr]))
        if token['name'] in self.svg_allow_local_href and (namespaces['xlink'], 'href') in attrs and re.search('^\\s*[^#\\s].*', attrs[namespaces['xlink'], 'href']):
            del attrs[namespaces['xlink'], 'href']
        if (None, 'style') in attrs:
            attrs[None, 'style'] = self.sanitize_css(attrs[None, 'style'])
        token['data'] = attrs
    return token