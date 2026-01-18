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
def set_manager_nic(self, nic_addr, nic_config):
    nic_info = self.get_manager_ethernet_uri(nic_addr)
    if nic_info.get('nic_addr') is None:
        return nic_info
    else:
        target_ethernet_uri = nic_info['nic_addr']
        target_ethernet_current_setting = nic_info['ethernet_setting']
    payload = {}
    for property in nic_config.keys():
        value = nic_config[property]
        if property in target_ethernet_current_setting and isinstance(value, dict) and isinstance(target_ethernet_current_setting[property], list):
            payload[property] = list()
            payload[property].append(value)
        else:
            payload[property] = value
    resp = self.patch_request(self.root_uri + target_ethernet_uri, payload, check_pyld=True)
    if resp['ret'] and resp['changed']:
        resp['msg'] = 'Modified manager NIC'
    return resp