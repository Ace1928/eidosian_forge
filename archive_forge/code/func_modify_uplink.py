from __future__ import (absolute_import, division, print_function)
import json
from ssl import SSLError
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.urls import ConnectionError, SSLValidationError
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.common.dict_transformations import recursive_diff
from ansible_collections.dellemc.openmanage.plugins.module_utils.ome import RestOME, ome_auth_params
from ansible_collections.dellemc.openmanage.plugins.module_utils.utils import get_item_and_list
def modify_uplink(module, rest_obj, fabric_id, uplink, uplinks):
    mparams = module.params
    pload_keys = ['Id', 'Name', 'Description', 'MediaType', 'NativeVLAN', 'UfdEnable', 'Ports', 'Networks']
    modify_payload = dict(((pload_key, uplink.get(pload_key)) for pload_key in pload_keys))
    port_list = list((port['Id'] for port in modify_payload['Ports']))
    modify_payload['Ports'] = sorted(list(set(port_list)))
    network_list = list((network['Id'] for network in modify_payload['Networks']))
    modify_payload['Networks'] = sorted(network_list)
    modify_data = {}
    if mparams.get('new_name'):
        modify_data['Name'] = mparams.get('new_name')
    if mparams.get('description'):
        modify_data['Description'] = mparams.get('description')
    if mparams.get('ufd_enable'):
        modify_data['UfdEnable'] = mparams.get('ufd_enable')
    if mparams.get('uplink_type'):
        if mparams.get('uplink_type') != uplink.get('MediaType'):
            module.fail_json(msg='Uplink Type cannot be modified.')
        modify_data['MediaType'] = mparams['uplink_type']
    if mparams.get('primary_switch_service_tag') or mparams.get('secondary_switch_service_tag'):
        if mparams.get('primary_switch_service_tag') == mparams.get('secondary_switch_service_tag'):
            module.fail_json(msg=SAME_SERVICE_TAG_MSG)
        payload_port_list = validate_ioms(module, rest_obj, uplinks)
        modify_data['Ports'] = sorted(list(set(payload_port_list)))
    media_id, mtypes = get_item_id(rest_obj, uplink.get('MediaType'), MEDIA_TYPES)
    if mparams.get('tagged_networks') and media_id:
        tagged_networks = validate_networks(module, rest_obj, fabric_id, media_id)
        modify_data['Networks'] = sorted(tagged_networks)
    if mparams.get('untagged_network') and media_id:
        untagged_id = validate_native_vlan(module, rest_obj, fabric_id, media_id)
        modify_data['NativeVLAN'] = untagged_id
    diff = recursive_diff(modify_data, modify_payload)
    if diff and diff[0]:
        modify_payload.update(diff[0])
        if module.check_mode:
            module.exit_json(changed=True, msg=CHECK_MODE_MSG)
        modify_payload['Ports'] = list(({'Id': port} for port in modify_payload['Ports']))
        modify_payload['Networks'] = list(({'Id': net} for net in modify_payload['Networks']))
        resp = rest_obj.invoke_request('PUT', UPLINK_URI.format(fabric_id=fabric_id, uplink_id=uplink['Id']), data=modify_payload)
        if isinstance(resp.json_data, dict):
            module.exit_json(changed=True, msg='Successfully modified the uplink.', uplink_id=uplink['Id'], additional_info=resp.json_data)
        module.exit_json(changed=True, msg='Successfully modified the uplink.', uplink_id=uplink['Id'])
    module.exit_json(msg=NO_CHANGES_MSG)