from __future__ import (absolute_import, division, print_function)
from collections import defaultdict
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible.module_utils.common.text.converters import to_native
from ansible_collections.community.routeros.plugins.module_utils.api import (
from ansible_collections.community.routeros.plugins.module_utils._api_data import (
def sync_single_value(module, api, path, path_info):
    data = module.params['data']
    if len(data) != 1:
        module.fail_json(msg='Data must be a list with exactly one element.')
    new_entry = data[0]
    polish_entry(new_entry, path_info, module, '')
    api_path = compose_api_path(api, path)
    old_data = get_api_data(api_path, path_info)
    if len(old_data) != 1:
        module.fail_json(msg='Internal error: retrieving /{path} resulted in {count} elements. Expected exactly 1.'.format(path=join_path(path), count=len(old_data)))
    old_entry = old_data[0]
    modifications, updated_entry = find_modifications(old_entry, new_entry, path_info, module, '')
    if modifications:
        if not module.check_mode:
            try:
                api_path.update(**modifications)
            except (LibRouterosError, UnicodeEncodeError) as e:
                module.fail_json(msg='Error while modifying: {error}'.format(error=to_native(e)))
            new_data = get_api_data(api_path, path_info)
            if len(new_data) == 1:
                updated_entry = new_data[0]
    remove_irrelevant_data(old_entry, path_info)
    remove_irrelevant_data(updated_entry, path_info)
    more = {}
    if module._diff:
        more['diff'] = {'before': old_entry, 'after': updated_entry}
    module.exit_json(changed=bool(modifications), old_data=[old_entry], new_data=[updated_entry], **more)