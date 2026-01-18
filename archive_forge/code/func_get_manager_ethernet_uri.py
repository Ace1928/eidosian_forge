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
def get_manager_ethernet_uri(self, nic_addr='null'):
    response = self.get_request(self.root_uri + self.manager_uri)
    if not response['ret']:
        return response
    data = response['data']
    if 'EthernetInterfaces' not in data:
        return {'ret': False, 'msg': 'EthernetInterfaces resource not found'}
    ethernetinterfaces_uri = data['EthernetInterfaces']['@odata.id']
    response = self.get_request(self.root_uri + ethernetinterfaces_uri)
    if not response['ret']:
        return response
    data = response['data']
    uris = [a.get('@odata.id') for a in data.get('Members', []) if a.get('@odata.id')]
    target_ethernet_uri = None
    target_ethernet_current_setting = None
    if nic_addr == 'null':
        nic_addr = self.root_uri.split('/')[-1]
        nic_addr = nic_addr.split(':')[0]
    for uri in uris:
        response = self.get_request(self.root_uri + uri)
        if not response['ret']:
            return response
        data = response['data']
        data_string = json.dumps(data)
        if nic_addr.lower() in data_string.lower():
            target_ethernet_uri = uri
            target_ethernet_current_setting = data
            break
    nic_info = {}
    nic_info['nic_addr'] = target_ethernet_uri
    nic_info['ethernet_setting'] = target_ethernet_current_setting
    if target_ethernet_uri is None:
        return {}
    else:
        return nic_info