from __future__ import (absolute_import, division, print_function)
from collections import defaultdict
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible.module_utils.common.text.converters import to_native
from ansible_collections.community.routeros.plugins.module_utils.api import (
from ansible_collections.community.routeros.plugins.module_utils._api_data import (
def prepare_for_add(entry, path_info):
    new_entry = {}
    for k, v in entry.items():
        if k.startswith('!'):
            real_k = k[1:]
            remove_value = path_info.fields[real_k].remove_value
            if remove_value is not None:
                k = real_k
                v = remove_value
        elif v is None:
            v = path_info.fields[k].remove_value
        new_entry[k] = v
    return new_entry