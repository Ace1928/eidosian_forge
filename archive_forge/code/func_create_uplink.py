from __future__ import (absolute_import, division, print_function)
import json
from ssl import SSLError
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.urls import ConnectionError, SSLValidationError
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.common.dict_transformations import recursive_diff
from ansible_collections.dellemc.openmanage.plugins.module_utils.ome import RestOME, ome_auth_params
from ansible_collections.dellemc.openmanage.plugins.module_utils.utils import get_item_and_list
def create_uplink(module, rest_obj, fabric_id, uplinks):
    mparams = module.params
    mandatory_parmas = ['name', 'uplink_type', 'tagged_networks']
    for prm in mandatory_parmas:
        if not mparams.get(prm):
            module.fail_json(msg='Mandatory parameter {0} not provided for uplink creation.'.format(prm))
    media_id, mtypes = get_item_id(rest_obj, mparams['uplink_type'], MEDIA_TYPES)
    if not media_id:
        module.fail_json(msg='Uplink Type {0} does not exist.'.format(mparams['uplink_type']))
    if mparams.get('primary_switch_service_tag') or mparams.get('secondary_switch_service_tag'):
        if mparams.get('primary_switch_service_tag') == mparams.get('secondary_switch_service_tag'):
            module.fail_json(msg=SAME_SERVICE_TAG_MSG)
        payload_port_list = validate_ioms(module, rest_obj, uplinks)
    else:
        module.fail_json(msg='Provide port details.')
    tagged_networks = validate_networks(module, rest_obj, fabric_id, media_id)
    create_payload = {'Name': mparams['name'], 'MediaType': mparams['uplink_type'], 'Ports': [{'Id': port} for port in payload_port_list], 'Networks': [{'Id': net} for net in tagged_networks]}
    if mparams.get('untagged_network'):
        untagged_id = validate_native_vlan(module, rest_obj, fabric_id, media_id)
        create_payload['NativeVLAN'] = untagged_id
    if mparams.get('ufd_enable'):
        create_payload['UfdEnable'] = mparams.get('ufd_enable')
    if mparams.get('description'):
        create_payload['Description'] = mparams.get('description')
    if module.check_mode:
        module.exit_json(changed=True, msg=CHECK_MODE_MSG)
    resp = rest_obj.invoke_request('POST', UPLINKS_URI.format(fabric_id=fabric_id), data=create_payload)
    uplink_id = resp.json_data
    if isinstance(resp.json_data, dict):
        uplink_id, tmp = get_item_id(rest_obj, mparams['name'], UPLINKS_URI.format(fabric_id=fabric_id))
        if not uplink_id:
            uplink_id = ''
        module.exit_json(changed=True, msg='Successfully created the uplink.', uplink_id=uplink_id, additional_info=resp.json_data)
    module.exit_json(changed=True, msg='Successfully created the uplink.', uplink_id=uplink_id)