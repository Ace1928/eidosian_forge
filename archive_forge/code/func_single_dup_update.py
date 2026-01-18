from __future__ import (absolute_import, division, print_function)
import json
from ssl import SSLError
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.openmanage.plugins.module_utils.ome import RestOME, ome_auth_params
from ansible.module_utils.urls import ConnectionError
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
def single_dup_update(rest_obj, module):
    target_data, device_ids, group_ids, baseline_ids = (None, None, None, None)
    if module.params.get('device_group_names') is not None:
        group_ids = get_group_ids(rest_obj, module)
    else:
        device_id_tags = _validate_device_attributes(module)
        device_ids, id_tag_map = get_device_ids(rest_obj, module, device_id_tags)
    if module.check_mode:
        module.exit_json(msg=CHANGES_FOUND, changed=True)
    upload_status, token = upload_dup_file(rest_obj, module)
    if upload_status:
        report_payload = get_dup_applicability_payload(token, device_ids=device_ids, group_ids=group_ids, baseline_ids=baseline_ids)
        if report_payload:
            target_data = get_applicable_components(rest_obj, report_payload, module)
    return target_data