from __future__ import (absolute_import, division, print_function)
import json
import re
import time
from ssl import SSLError
from ansible_collections.dellemc.openmanage.plugins.module_utils.redfish import Redfish, redfish_auth_params, \
from ansible_collections.dellemc.openmanage.plugins.module_utils.utils import wait_for_redfish_reboot_job, \
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.urls import ConnectionError, SSLValidationError
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
def get_rollback_preview_target(redfish_obj, module):
    action_resp = redfish_obj.invoke_request('GET', '{0}{1}'.format(redfish_obj.root_uri, UPDATE_SERVICE))
    action_attr = action_resp.json_data['Actions']
    update_uri = None
    if '#UpdateService.SimpleUpdate' in action_attr:
        update_service = action_attr.get('#UpdateService.SimpleUpdate')
        if 'target' not in update_service:
            module.fail_json(msg=NOT_SUPPORTED)
        update_uri = update_service.get('target')
    inventory_uri = action_resp.json_data.get('FirmwareInventory').get('@odata.id')
    inventory_uri_resp = redfish_obj.invoke_request('GET', '{0}{1}'.format(inventory_uri, '?$expand=*($levels=1)'), api_timeout=120)
    previous_component = list(filter(lambda d: d['Id'].startswith('Previous'), inventory_uri_resp.json_data['Members']))
    if not previous_component:
        module.fail_json(msg=NO_COMPONENTS)
    component_name = module.params['name']
    try:
        component_compile = re.compile('^{0}$'.format(component_name))
    except Exception:
        module.exit_json(msg=NO_CHANGES_FOUND)
    prev_uri, reboot_uri = ({}, [])
    for each in previous_component:
        available_comp = each['Name']
        available_name = re.match(component_compile, available_comp)
        if not available_name:
            continue
        if available_name.group() in REBOOT_COMP:
            reboot_uri.append(each['@odata.id'])
            continue
        prev_uri[each['Version']] = each['@odata.id']
    if module.check_mode and (prev_uri or reboot_uri):
        module.exit_json(msg=CHANGES_FOUND, changed=True)
    elif not prev_uri and (not reboot_uri):
        module.exit_json(msg=NO_CHANGES_FOUND)
    return (list(prev_uri.values()), reboot_uri, update_uri)