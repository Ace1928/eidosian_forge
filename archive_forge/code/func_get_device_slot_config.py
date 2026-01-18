from __future__ import (absolute_import, division, print_function)
import json
import time
from ssl import SSLError
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.urls import ConnectionError
from ansible_collections.dellemc.openmanage.plugins.module_utils.ome import RestOME, ome_auth_params
from ansible.module_utils.common.dict_transformations import recursive_diff
def get_device_slot_config(module, rest_obj):
    ids, tags = ({}, {})
    dvc_list = []
    for dvc in module.params.get('device_options'):
        sn = dvc.get('slot_name')
        id = dvc.get('device_id')
        st = dvc.get('device_service_tag')
        if id:
            ids[str(id)] = sn
            dvc_list.append(str(id))
        else:
            tags[st] = sn
            dvc_list.append(st)
    duplicate = [x for i, x in enumerate(dvc_list) if i != dvc_list.index(x)]
    if duplicate:
        module.fail_json(msg=DEVICE_REPEATED.format(';'.join(set(duplicate))))
    resp = rest_obj.get_all_items_with_pagination(DEVICE_URI)
    devices = resp.get('value')
    all_dvcs = {}
    invalid_slots = set()
    ident_map, name_map = ({}, {})
    for dvc in devices:
        if not ids and (not tags):
            break
        id = str(dvc.get('Id'))
        tag = dvc.get('Identifier')
        slot_cfg = dvc.get('SlotConfiguration')
        all_dvcs[tag] = slot_cfg
        if id in ids:
            if not slot_cfg or not slot_cfg.get('SlotNumber'):
                invalid_slots.add(id)
            else:
                ident_map[id] = tag
                name_map[id] = slot_cfg['SlotName']
                slot_cfg['new_name'] = ids[id]
                slot_cfg['DeviceServiceTag'] = tag
                slot_cfg['DeviceId'] = id
        if tag in tags:
            if not slot_cfg or not slot_cfg.get('SlotNumber'):
                invalid_slots.add(tag)
            else:
                ident_map[tag] = tag
                name_map[tag] = slot_cfg['SlotName']
                slot_cfg['new_name'] = tags[tag]
                slot_cfg['DeviceServiceTag'] = tag
                slot_cfg['DeviceId'] = id
    idf_list = list(ident_map.values())
    duplicate = [x for i, x in enumerate(idf_list) if i != idf_list.index(x)]
    if duplicate:
        module.fail_json(msg=DEVICE_REPEATED.format(';'.join(set(duplicate))))
    invalid_slots.update(set(ids.keys()) - set(ident_map.keys()))
    invalid_slots.update(set(tags.keys()) - set(ident_map.keys()))
    if invalid_slots:
        module.fail_json(msg=INVALID_SLOT_DEVICE.format(';'.join(invalid_slots)))
    slot_dict_diff = {}
    id_diff = recursive_diff(ids, name_map)
    if id_diff and id_diff[0]:
        diff = dict([(int(k), all_dvcs[ident_map[k]]) for k, v in id_diff[0].items()])
        slot_dict_diff.update(diff)
    tag_diff = recursive_diff(tags, name_map)
    if tag_diff and tag_diff[0]:
        diff = dict([(ident_map[k], all_dvcs[k]) for k, v in tag_diff[0].items()])
        slot_dict_diff.update(diff)
    if not slot_dict_diff:
        module.exit_json(msg=NO_CHANGES_MSG)
    if module.check_mode:
        module.exit_json(msg=CHANGES_FOUND, changed=True)
    return slot_dict_diff