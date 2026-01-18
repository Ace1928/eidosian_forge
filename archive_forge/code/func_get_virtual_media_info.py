from __future__ import (absolute_import, division, print_function)
import json
import copy
import time
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.urls import ConnectionError, SSLValidationError
from ansible_collections.dellemc.openmanage.plugins.module_utils.idrac_redfish import iDRACRedfishAPI, idrac_auth_params
from ansible.module_utils.basic import AnsibleModule
def get_virtual_media_info(idrac):
    resp = idrac.invoke_request('/redfish/v1/', 'GET')
    redfish_version = resp.json_data['RedfishVersion']
    rd_version = redfish_version.replace('.', '')
    if 1131 <= int(rd_version):
        vr_id = 'system'
        member_resp = idrac.invoke_request('{0}?$expand=*($levels=1)'.format(SYSTEM_BASE), 'GET')
    else:
        vr_id = 'manager'
        member_resp = idrac.invoke_request('{0}?$expand=*($levels=1)'.format(MANAGER_BASE), 'GET')
    response = member_resp.json_data['Members']
    return (response, vr_id, rd_version)