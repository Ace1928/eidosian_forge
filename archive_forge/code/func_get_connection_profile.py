from __future__ import (absolute_import, division, print_function)
import json
import time
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.openmanage.plugins.module_utils.ome import RestOME, ome_auth_params
from ansible_collections.dellemc.openmanage.plugins.module_utils.utils import strip_substr_dict
from ansible.module_utils.urls import ConnectionError, SSLValidationError
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.common.dict_transformations import snake_dict_to_camel_dict
def get_connection_profile(disc_config):
    proto_add_dict = {'wsman': {'certificateDetail': None, 'isHttp': False, 'keepAlive': True}, 'redfish': {'certificateDetail': None, 'isHttp': False, 'keepAlive': True}, 'snmp': {'enableV1V2': True, 'enableV3': False}, 'vmware': {'certificateDetail': None, 'isHttp': False, 'keepAlive': False}, 'ssh': {'useKey': False, 'key': None, 'knownHostKey': None, 'passphrase': None}, 'ipmi': {'privilege': 2}, 'storage': {'certificateDetail': None, 'isHttp': False, 'keepAlive': True}}
    proto_list = ['wsman', 'snmp', 'vmware', 'ssh', 'ipmi', 'redfish', 'storage']
    conn_profile = {'profileId': 0, 'profileName': '', 'profileDescription': '', 'type': 'DISCOVERY'}
    creds_dict = {}
    for p in proto_list:
        if disc_config.get(p):
            xproto = {'type': p.upper(), 'authType': 'Basic', 'modified': False}
            xproto['credentials'] = snake_dict_to_camel_dict(disc_config[p])
            xproto['credentials'].update(proto_add_dict.get(p, {}))
            creds_dict[p] = xproto
            if p == 'wsman':
                rf = xproto.copy()
                rf['type'] = 'REDFISH'
                creds_dict['redfish'] = rf
    conn_profile['credentials'] = list(creds_dict.values())
    return conn_profile