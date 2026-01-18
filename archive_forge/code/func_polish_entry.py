from __future__ import (absolute_import, division, print_function)
from collections import defaultdict
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible.module_utils.common.text.converters import to_native
from ansible_collections.community.routeros.plugins.module_utils.api import (
from ansible_collections.community.routeros.plugins.module_utils._api_data import (
def polish_entry(entry, path_info, module, for_text):
    if '.id' in entry:
        entry.pop('.id')
    to_remove = []
    for key, value in entry.items():
        real_key = key
        disabled_key = False
        if key.startswith('!'):
            disabled_key = True
            key = key[1:]
            if key in entry:
                module.fail_json(msg='Not both "{key}" and "!{key}" must appear{for_text}.'.format(key=key, for_text=for_text))
        key_info = path_info.fields.get(key)
        if key_info is None:
            module.fail_json(msg='Unknown key "{key}"{for_text}.'.format(key=real_key, for_text=for_text))
        if disabled_key:
            if not key_info.can_disable:
                module.fail_json(msg='Key "!{key}" must not be disabled (leading "!"){for_text}.'.format(key=key, for_text=for_text))
            if value not in (None, '', key_info.remove_value):
                module.fail_json(msg='Disabled key "!{key}" must not have a value{for_text}.'.format(key=key, for_text=for_text))
        elif value is None:
            if not key_info.can_disable:
                module.fail_json(msg='Key "{key}" must not be disabled (value null/~/None){for_text}.'.format(key=key, for_text=for_text))
        if key_info.read_only:
            if module.params['handle_read_only'] == 'error':
                module.fail_json(msg='Key "{key}" is read-only{for_text}, and handle_read_only=error.'.format(key=key, for_text=for_text))
            if module.params['handle_read_only'] == 'ignore':
                to_remove.append(real_key)
        if key_info.write_only:
            if module.params['handle_write_only'] == 'error':
                module.fail_json(msg='Key "{key}" is write-only{for_text}, and handle_write_only=error.'.format(key=key, for_text=for_text))
    for key in to_remove:
        entry.pop(key)
    for key, field_info in path_info.fields.items():
        if field_info.required and key not in entry:
            module.fail_json(msg='Key "{key}" must be present{for_text}.'.format(key=key, for_text=for_text))
    for require_list in path_info.required_one_of:
        found_req_keys = [rk for rk in require_list if rk in entry]
        if len(require_list) > 0 and (not found_req_keys):
            module.fail_json(msg='Every element in data must contain one of {required_keys}. For example, the element{for_text} does not provide it.'.format(required_keys=', '.join(['"{k}"'.format(k=k) for k in require_list]), for_text=for_text))
    for exclusive_list in path_info.mutually_exclusive:
        found_ex_keys = [ek for ek in exclusive_list if ek in entry]
        if len(found_ex_keys) > 1:
            module.fail_json(msg='Keys {exclusive_keys} cannot be used at the same time{for_text}.'.format(exclusive_keys=', '.join(['"{k}"'.format(k=k) for k in found_ex_keys]), for_text=for_text))