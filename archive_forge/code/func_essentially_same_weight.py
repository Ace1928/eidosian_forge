from __future__ import (absolute_import, division, print_function)
from collections import defaultdict
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible.module_utils.common.text.converters import to_native
from ansible_collections.community.routeros.plugins.module_utils.api import (
from ansible_collections.community.routeros.plugins.module_utils._api_data import (
def essentially_same_weight(old_entry, new_entry, path_info, module):
    for k, v in new_entry.items():
        if k == '.id':
            continue
        disabled_k = None
        if k.startswith('!'):
            disabled_k = k[1:]
        elif v is None or v == path_info.fields[k].remove_value:
            disabled_k = k
        if disabled_k is not None:
            if disabled_k in old_entry:
                return None
            continue
        if k not in old_entry and path_info.fields[k].default == v:
            continue
        if k not in old_entry or old_entry[k] != v:
            return None
    handle_entries_content = module.params['handle_entries_content']
    weight = 0
    for k in old_entry:
        if k == '.id' or k in new_entry or '!%s' % k in new_entry or (k not in path_info.fields):
            continue
        field_info = path_info.fields[k]
        if field_info.default is not None and field_info.default == old_entry[k]:
            continue
        if handle_entries_content != 'ignore':
            return None
        else:
            weight += 1
    return weight