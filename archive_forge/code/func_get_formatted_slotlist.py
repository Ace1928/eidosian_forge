from __future__ import (absolute_import, division, print_function)
import json
import time
from ssl import SSLError
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.urls import ConnectionError
from ansible_collections.dellemc.openmanage.plugins.module_utils.ome import RestOME, ome_auth_params
from ansible.module_utils.common.dict_transformations import recursive_diff
def get_formatted_slotlist(slot_dict):
    slot_list = list(slot_dict.values())
    req_tup = ('slot', 'job', 'chassis', 'device')
    for slot in slot_list:
        cp = slot.copy()
        klist = cp.keys()
        for k in klist:
            if not str(k).lower().startswith(req_tup):
                slot.pop(k)
    return slot_list