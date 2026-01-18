from __future__ import absolute_import, division, print_function
from ansible.module_utils.six.moves.urllib.parse import quote
from . import errors
def build_core_v2_path(namespace, *parts):
    return build_url_path('core', 'v2', namespace, *parts)