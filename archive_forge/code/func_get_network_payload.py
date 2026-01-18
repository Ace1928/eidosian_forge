from __future__ import (absolute_import, division, print_function)
import json
import socket
import copy
from ssl import SSLError
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.urls import ConnectionError
from ansible_collections.dellemc.openmanage.plugins.module_utils.ome import RestOME, ome_auth_params
from ansible.module_utils.common.dict_transformations import recursive_diff
def get_network_payload(module, rest_obj, dvc):
    resp = rest_obj.invoke_request('GET', NETWORK_SETTINGS.format(dvc.get('Id')))
    got_payload = resp.json_data
    payload = rest_obj.strip_substr_dict(got_payload)
    update_dict = {CHASSIS: update_chassis_payload, SERVER: update_server_payload, IO_MODULE: update_iom_payload}
    diff = update_dict[dvc.get('Type')](module, payload)
    if not diff:
        module.exit_json(msg=NO_CHANGES_MSG)
    if module.check_mode:
        module.exit_json(msg=CHANGES_FOUND, changed=True)
    return payload