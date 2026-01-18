from __future__ import absolute_import, unicode_literals
import datetime
import os
from oauthlib.common import unicode_type, urldecode
def list_to_scope(scope):
    """Convert a list of scopes to a space separated string."""
    if isinstance(scope, unicode_type) or scope is None:
        return scope
    elif isinstance(scope, (set, tuple, list)):
        return ' '.join([unicode_type(s) for s in scope])
    else:
        raise ValueError('Invalid scope (%s), must be string, tuple, set, or list.' % scope)