from __future__ import (absolute_import, division, print_function)
import json
from ssl import SSLError
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.openmanage.plugins.module_utils.ome import RestOME, ome_auth_params
from ansible.module_utils.urls import ConnectionError, SSLValidationError
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
def modify_vlan(module, rest_obj, vlan_id, vlans):
    payload = format_payload(module.params)
    payload['Description'] = module.params.get('description')
    if module.params.get('type'):
        payload['Type'], types = get_item_id(rest_obj, module.params['type'], VLAN_TYPES)
        if not payload['Type']:
            module.fail_json(msg="Network type '{0}' not found.".format(module.params['type']))
    if module.params.get('new_name'):
        payload['Name'] = module.params['new_name']
    current_setting = {}
    for i in range(len(vlans)):
        if vlans[i]['Id'] == vlan_id:
            current_setting = vlans.pop(i)
            break
    diff = 0
    for config, pload in payload.items():
        pval = payload.get(config)
        if pval is not None:
            if current_setting.get(config) != pval:
                payload[config] = pval
                diff += 1
        else:
            payload[config] = current_setting.get(config)
    if payload['VlanMinimum'] > payload['VlanMaximum']:
        module.fail_json(msg=VLAN_VALUE_MSG)
    overlap = check_overlapping_vlan_range(payload, vlans)
    if overlap:
        module.fail_json(msg=VLAN_RANGE_OVERLAP.format(vlan_name=overlap['Name'], vlan_min=overlap['VlanMinimum'], vlan_max=overlap['VlanMaximum']))
    if diff == 0:
        if module.check_mode:
            module.exit_json(msg='No changes found to be applied to the VLAN configuration.')
        module.exit_json(msg='No changes found to be applied as the entered values are the same as the current configuration.', vlan_status=current_setting)
    if module.check_mode:
        module.exit_json(changed=True, msg=CHECK_MODE_MSG)
    payload['Id'] = vlan_id
    resp = rest_obj.invoke_request('PUT', VLAN_ID_CONFIG.format(Id=vlan_id), data=payload)
    module.exit_json(msg='Successfully updated the VLAN.', vlan_status=resp.json_data, changed=True)