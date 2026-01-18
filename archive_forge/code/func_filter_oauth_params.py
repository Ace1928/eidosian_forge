from __future__ import absolute_import, unicode_literals
from oauthlib.common import quote, unicode_type, unquote
def filter_oauth_params(params):
    """Removes all non oauth parameters from a dict or a list of params."""
    is_oauth = lambda kv: kv[0].startswith('oauth_')
    if isinstance(params, dict):
        return list(filter(is_oauth, list(params.items())))
    else:
        return list(filter(is_oauth, params))