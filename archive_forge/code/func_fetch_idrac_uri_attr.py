from __future__ import (absolute_import, division, print_function)
import json
import re
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.urls import ConnectionError
from ansible_collections.dellemc.openmanage.plugins.module_utils.idrac_redfish import iDRACRedfishAPI, idrac_auth_params
from ansible_collections.dellemc.openmanage.plugins.module_utils.utils import get_manager_res_id
from ansible.module_utils.basic import AnsibleModule
def fetch_idrac_uri_attr(idrac, module, res_id):
    diff = 0
    uri_dict = {}
    idrac_response_attr = {}
    system_response_attr = {}
    lc_response_attr = {}
    response = idrac.invoke_request('{0}/{1}'.format(MANAGERS_URI, res_id), 'GET')
    dell_attributes = response.json_data.get('Links', {}).get('Oem', {}).get('Dell', {}).get('DellAttributes')
    if dell_attributes:
        for item in dell_attributes:
            uri = item.get('@odata.id')
            attr_id = uri.split('/')[-1]
            uri_dict[attr_id] = uri
        idrac_attr = module.params.get('idrac_attributes')
        system_attr = module.params.get('system_attributes')
        lc_attr = module.params.get('lifecycle_controller_attributes')
        invalid = {}
        attr_registry = get_attributes_registry(idrac)
        if idrac_attr is not None:
            x, idrac_response_attr = get_response_attr(idrac, MANAGER_ID, idrac_attr, uri_dict)
            invalid.update(validate_vs_registry(attr_registry, idrac_response_attr))
            diff += x
        if system_attr is not None:
            x, system_response_attr = get_response_attr(idrac, SYSTEM_ID, system_attr, uri_dict)
            invalid.update(validate_vs_registry(attr_registry, system_response_attr))
            diff += x
        if lc_attr is not None:
            x, lc_response_attr = get_response_attr(idrac, LC_ID, lc_attr, uri_dict)
            invalid.update(validate_vs_registry(attr_registry, lc_response_attr))
            diff += x
        if invalid:
            module.exit_json(failed=True, msg='Attributes have invalid values.', invalid_attributes=invalid)
    else:
        job_resp = scp_idrac_attributes(module, idrac, res_id)
        if job_resp.status_code == 200:
            error_msg = ['Unable to complete application of configuration profile values.', 'Import of Server Configuration Profile operation completed with errors.']
            message = job_resp.json_data['Message']
            message_id = job_resp.json_data['MessageId']
            if message_id == 'SYS069':
                module.exit_json(msg=NO_CHANGES_MSG)
            elif message_id == 'SYS053':
                module.exit_json(msg=SUCCESS_MSG, changed=True)
            elif message in error_msg:
                module.fail_json(msg=ATTR_FAIL_MSG)
            else:
                module.fail_json(msg=message)
    return (diff, uri_dict, idrac_response_attr, system_response_attr, lc_response_attr)