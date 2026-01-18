from __future__ import (absolute_import, division, print_function)
import json
import time
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.urls import ConnectionError, SSLValidationError
from ansible_collections.dellemc.openmanage.plugins.module_utils.idrac_redfish import iDRACRedfishAPI, idrac_auth_params
from ansible_collections.dellemc.openmanage.plugins.module_utils.utils import (strip_substr_dict, idrac_system_reset,
from ansible.module_utils.basic import AnsibleModule
def get_response_attributes(module, idrac, res_id):
    resp = idrac.invoke_request('{0}/{1}'.format(SYSTEM_URI, res_id), 'GET')
    resp_data = resp.json_data['Boot']
    resp_data.pop('Certificates', None)
    resp_data.pop('BootOrder@odata.count', None)
    resp_data.pop('BootSourceOverrideTarget@Redfish.AllowableValues', None)
    if resp_data.get('BootOptions') is None and module.params.get('boot_options') is not None:
        module.fail_json(msg=UNSUPPORTED_MSG)
    if resp.json_data.get('Actions') is not None:
        type_reset = resp.json_data['Actions']['#ComputerSystem.Reset']['ResetType@Redfish.AllowableValues']
        if 'GracefulRestart' not in type_reset:
            RESET_TYPE['graceful_restart'] = 'ForceRestart'
    return resp_data