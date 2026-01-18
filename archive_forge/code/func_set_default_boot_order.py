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
def set_default_boot_order(self):
    systems_uri = self.systems_uri
    response = self.get_request(self.root_uri + systems_uri)
    if response['ret'] is False:
        return response
    data = response['data']
    action = '#ComputerSystem.SetDefaultBootOrder'
    if 'Actions' not in data or action not in data['Actions']:
        return {'ret': False, 'msg': 'Action %s not found' % action}
    if 'target' not in data['Actions'][action]:
        return {'ret': False, 'msg': 'target URI missing from Action %s' % action}
    action_uri = data['Actions'][action]['target']
    payload = {}
    response = self.post_request(self.root_uri + action_uri, payload)
    if response['ret'] is False:
        return response
    return {'ret': True, 'changed': True, 'msg': 'BootOrder set to default'}