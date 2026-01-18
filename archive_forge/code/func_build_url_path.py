from __future__ import absolute_import, division, print_function
from ansible.module_utils.six.moves.urllib.parse import quote
from . import errors
def build_url_path(api_group, api_version, namespace, *parts):
    prefix = '/api/{0}/{1}/'.format(api_group, api_version)
    if namespace:
        prefix += 'namespaces/{0}/'.format(quote(namespace, safe=''))
    return prefix + '/'.join((quote(p, safe='') for p in parts if p))