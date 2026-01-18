from __future__ import absolute_import, division, print_function
import json
import os
import random
import string
import gzip
from io import BytesIO
from ansible.module_utils.urls import open_url
from ansible.module_utils.common.text.converters import to_native
from ansible.module_utils.common.text.converters import to_text
from ansible.module_utils.common.text.converters import to_bytes
from ansible.module_utils.six import text_type
from ansible.module_utils.six.moves import http_client
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.six.moves.urllib.parse import urlparse
from ansible.module_utils.ansible_release import __version__ as ansible_version
from ansible_collections.community.general.plugins.module_utils.version import LooseVersion
def clear_sessions(self):
    response = self.get_request(self.root_uri + self.sessions_uri)
    if response['ret'] is False:
        return response
    data = response['data']
    if data['Members@odata.count'] == 0:
        return {'ret': True, 'changed': False, 'msg': 'There are no active sessions'}
    for session in data[u'Members']:
        response = self.delete_request(self.root_uri + session[u'@odata.id'])
        if response['ret'] is False:
            return response
    return {'ret': True, 'changed': True, 'msg': 'Cleared all sessions successfully'}