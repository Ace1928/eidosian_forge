from __future__ import (absolute_import, division, print_function)
import json
from ssl import SSLError
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.urls import ConnectionError
from ansible_collections.dellemc.openmanage.plugins.module_utils.ome import RestOME, ome_auth_params
from ansible.module_utils.common.dict_transformations import recursive_diff
from ansible.module_utils.common.dict_transformations import snake_dict_to_camel_dict
def compare_get_payload(module, current_list, input_config):
    payload_list = [strip_substr_dict(sys) for sys in current_list]
    current_config = dict([(sys.get('Id'), sys) for sys in payload_list])
    diff = 0
    for k, v in current_config.items():
        i_dict = input_config.get(k)
        if i_dict:
            d = recursive_diff(i_dict, v)
            if d and d[0]:
                v.update(d[0])
                diff = diff + 1
        v.pop('Id', None)
        payload_list[int(k) - 1] = v
    if not diff:
        module.exit_json(msg=NO_CHANGES_MSG)
    if module.check_mode:
        module.exit_json(msg=CHANGES_FOUND, changed=True)
    return payload_list