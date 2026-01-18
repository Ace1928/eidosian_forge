from __future__ import (absolute_import, division, print_function)
import json
import time
from ssl import SSLError
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.openmanage.plugins.module_utils.ome import RestOME, ome_auth_params
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.urls import ConnectionError, SSLValidationError
from ansible.module_utils.common.dict_transformations import recursive_diff
def get_network_iso_payload(module):
    boot_iso_dict = module.params.get('boot_to_network_iso')
    iso_payload = {}
    if boot_iso_dict:
        iso_payload = {'BootToNetwork': False}
        if boot_iso_dict.get('boot_to_network'):
            iso_payload['BootToNetwork'] = True
            share_type = boot_iso_dict.get('share_type')
            iso_payload['ShareType'] = share_type
            share_detail = {}
            sh_ip = boot_iso_dict.get('share_ip')
            share_detail['IpAddress'] = sh_ip
            share_detail['ShareName'] = sh_ip
            share_detail['User'] = boot_iso_dict.get('share_user')
            share_detail['Password'] = boot_iso_dict.get('share_password')
            share_detail['WorkGroup'] = boot_iso_dict.get('workgroup')
            iso_payload['ShareDetail'] = share_detail
            if str(boot_iso_dict.get('iso_path')).lower().endswith('.iso'):
                iso_payload['IsoPath'] = boot_iso_dict.get('iso_path')
            else:
                module.fail_json(msg="ISO path does not have extension '.iso'")
            iso_payload['IsoTimeout'] = boot_iso_dict.get('iso_timeout')
    return iso_payload