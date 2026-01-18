from __future__ import absolute_import, division, print_function
from ansible.module_utils.six.moves.urllib.parse import quote
from . import errors
def convert_v1_to_v2_response(response):
    if not response:
        return response
    if 'metadata' not in response:
        return response['spec']
    return dict(response['spec'], metadata=response['metadata'])