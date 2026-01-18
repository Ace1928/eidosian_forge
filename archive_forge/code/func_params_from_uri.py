from __future__ import absolute_import, unicode_literals
import datetime
import os
from oauthlib.common import unicode_type, urldecode
def params_from_uri(uri):
    params = dict(urldecode(urlparse(uri).query))
    if 'scope' in params:
        params['scope'] = scope_to_list(params['scope'])
    return params