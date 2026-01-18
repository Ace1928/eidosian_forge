from __future__ import absolute_import, unicode_literals
from oauthlib.common import quote, unicode_type, unquote
def parse_authorization_header(authorization_header):
    """Parse an OAuth authorization header into a list of 2-tuples"""
    auth_scheme = 'OAuth '.lower()
    if authorization_header[:len(auth_scheme)].lower().startswith(auth_scheme):
        items = parse_http_list(authorization_header[len(auth_scheme):])
        try:
            return list(parse_keqv_list(items).items())
        except (IndexError, ValueError):
            pass
    raise ValueError('Malformed authorization header')