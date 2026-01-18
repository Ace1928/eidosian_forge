from __future__ import (absolute_import, division, print_function)
import re
import json
import codecs
import binascii
from ssl import SSLError
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.openmanage.plugins.module_utils.ome import RestOME, ome_auth_params
from ansible.module_utils.urls import ConnectionError
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
def get_updated_modify_payload(modify_payload, existing_payload):
    """update current pool setting setting to modify payload if not provided
     in the options to avoid the null update from ome"""
    remove_unwanted_key_list = ['@odata.type', '@odata.id', 'CreatedBy', 'CreationTime', 'LastUpdatedBy', 'LastUpdateTime', 'UsageCounts', 'UsageIdentitySets@odata.navigationLink']
    [existing_payload.pop(key) for key in remove_unwanted_key_list if key in existing_payload]
    for key, val in existing_payload.items():
        if key not in modify_payload:
            modify_payload[key] = val
        elif existing_payload.get(key) and key == 'EthernetSettings' or key == 'FcoeSettings':
            update_modify_setting(modify_payload, existing_payload, key, ['Mac'])
        elif existing_payload.get(key) and key == 'FcSettings':
            update_modify_setting(modify_payload, existing_payload, key, ['Wwnn', 'Wwpn'])
        elif existing_payload.get(key) and key == 'IscsiSettings':
            update_modify_setting(modify_payload, existing_payload, key, ['Mac', 'InitiatorConfig', 'InitiatorIpPoolSettings'])
    modify_payload = dict([(k, v) for k, v in modify_payload.items() if v is not None])
    return modify_payload