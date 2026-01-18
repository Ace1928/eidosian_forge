from __future__ import (absolute_import, division, print_function)
import json
from ssl import SSLError
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.openmanage.plugins.module_utils.ome import RestOME, ome_auth_params
from ansible.module_utils.urls import ConnectionError, SSLValidationError
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
def create_vlan(module, rest_obj, vlans):
    payload = format_payload(module.params)
    if not all(payload.values()):
        module.fail_json(msg='The vlan_minimum, vlan_maximum and type values are required for creating a VLAN.')
    if payload['VlanMinimum'] > payload['VlanMaximum']:
        module.fail_json(msg=VLAN_VALUE_MSG)
    overlap = check_overlapping_vlan_range(payload, vlans)
    if overlap:
        module.fail_json(msg=VLAN_RANGE_OVERLAP.format(vlan_name=overlap['Name'], vlan_min=overlap['VlanMinimum'], vlan_max=overlap['VlanMaximum']))
    if module.check_mode:
        module.exit_json(changed=True, msg=CHECK_MODE_MSG)
    if module.params.get('description'):
        payload['Description'] = module.params.get('description')
    payload['Type'], types = get_item_id(rest_obj, module.params['type'], VLAN_TYPES)
    if not payload['Type']:
        module.fail_json(msg="Network type '{0}' not found.".format(module.params['type']))
    resp = rest_obj.invoke_request('POST', VLAN_CONFIG, data=payload)
    module.exit_json(msg='Successfully created the VLAN.', vlan_status=resp.json_data, changed=True)