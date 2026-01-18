from __future__ import (absolute_import, division, print_function)
import json
from ssl import SSLError
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.urls import ConnectionError
from ansible_collections.dellemc.openmanage.plugins.module_utils.ome import RestOME, ome_auth_params
from ansible_collections.dellemc.openmanage.plugins.module_utils.utils import strip_substr_dict
def create_cifs_payload(module, curr_payload):
    console_setting_list = []
    updated_payload = {'ConsoleSetting': []}
    payload_dict = create_payload_dict(curr_payload)
    get_bas = module.params.get('builtin_appliance_share')
    if get_bas and get_bas.get('share_options'):
        payload = payload_dict['SHARE_TYPE'].copy()
        payload['Value'] = get_bas.get('share_options')
        console_setting_list.append(payload)
    updated_payload['ConsoleSetting'] = console_setting_list
    return updated_payload