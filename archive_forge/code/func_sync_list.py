from __future__ import (absolute_import, division, print_function)
from collections import defaultdict
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible.module_utils.common.text.converters import to_native
from ansible_collections.community.routeros.plugins.module_utils.api import (
from ansible_collections.community.routeros.plugins.module_utils._api_data import (
def sync_list(module, api, path, path_info):
    handle_absent_entries = module.params['handle_absent_entries']
    handle_entries_content = module.params['handle_entries_content']
    if handle_absent_entries == 'remove':
        if handle_entries_content == 'ignore':
            module.fail_json('For this path, handle_absent_entries=remove cannot be combined with handle_entries_content=ignore')
    stratify_keys = path_info.stratify_keys or ()
    data = module.params['data']
    stratified_data = defaultdict(list)
    for index, entry in enumerate(data):
        for stratify_key in stratify_keys:
            if stratify_key not in entry:
                module.fail_json(msg='Every element in data must contain "{stratify_key}". For example, the element at index #{index} does not provide it.'.format(stratify_key=stratify_key, index=index + 1))
        sks = tuple((entry[stratify_key] for stratify_key in stratify_keys))
        polish_entry(entry, path_info, module, ' at index {index}'.format(index=index + 1))
        stratified_data[sks].append((index, entry))
    stratified_data = dict(stratified_data)
    api_path = compose_api_path(api, path)
    old_data = get_api_data(api_path, path_info)
    old_data = remove_dynamic(old_data)
    stratified_old_data = defaultdict(list)
    for index, entry in enumerate(old_data):
        sks = tuple((entry[stratify_key] for stratify_key in stratify_keys))
        stratified_old_data[sks].append((index, entry))
    stratified_old_data = dict(stratified_old_data)
    create_list = []
    modify_list = []
    remove_list = []
    new_data = []
    for key, indexed_entries in stratified_data.items():
        old_entries = stratified_old_data.pop(key, [])
        matching_old_entries, unmatched_old_entries = match_entries(indexed_entries, old_entries, path_info, module)
        for (index, new_entry), potential_old_entry in zip(indexed_entries, matching_old_entries):
            if potential_old_entry is not None:
                old_index, old_entry = potential_old_entry
                modifications, updated_entry = find_modifications(old_entry, new_entry, path_info, module, ' at index {index}'.format(index=index + 1))
                if modifications:
                    modifications['.id'] = old_entry['.id']
                    modify_list.append(modifications)
                new_data.append((old_index, updated_entry))
                new_entry['.id'] = old_entry['.id']
            else:
                remove_read_only(new_entry, path_info)
                create_list.append(new_entry)
        if handle_absent_entries == 'remove':
            remove_list.extend((entry['.id'] for index, entry in unmatched_old_entries))
        else:
            new_data.extend(unmatched_old_entries)
    for key, entries in stratified_old_data.items():
        if handle_absent_entries == 'remove':
            remove_list.extend((entry['.id'] for index, entry in entries))
        else:
            new_data.extend(entries)
    new_data = [entry for index, entry in sorted(new_data, key=lambda entry: entry[0])]
    new_data.extend(create_list)
    reorder_list = []
    if module.params['ensure_order']:
        for index, entry in enumerate(data):
            if '.id' in entry:

                def match(current_entry):
                    return current_entry['.id'] == entry['.id']
            else:

                def match(current_entry):
                    return current_entry is entry
            current_index = next((current_index + index for current_index, current_entry in enumerate(new_data[index:]) if match(current_entry)))
            if current_index != index:
                reorder_list.append((index, new_data[current_index], new_data[index]))
                new_data.insert(index, new_data.pop(current_index))
    if not module.check_mode:
        if remove_list:
            try:
                api_path.remove(*remove_list)
            except (LibRouterosError, UnicodeEncodeError) as e:
                module.fail_json(msg='Error while removing {remove_list}: {error}'.format(remove_list=', '.join(['ID {id}'.format(id=id) for id in remove_list]), error=to_native(e)))
        for modifications in modify_list:
            try:
                api_path.update(**modifications)
            except (LibRouterosError, UnicodeEncodeError) as e:
                module.fail_json(msg='Error while modifying for ID {id}: {error}'.format(id=modifications['.id'], error=to_native(e)))
        for entry in create_list:
            try:
                entry['.id'] = api_path.add(**prepare_for_add(entry, path_info))
            except (LibRouterosError, UnicodeEncodeError) as e:
                module.fail_json(msg='Error while creating entry: {error}'.format(error=to_native(e)))
        for new_index, new_entry, old_entry in reorder_list:
            try:
                for res in api_path('move', numbers=new_entry['.id'], destination=old_entry['.id']):
                    pass
            except (LibRouterosError, UnicodeEncodeError) as e:
                module.fail_json(msg='Error while moving entry ID {element_id} to position #{new_index} ID ({new_id}): {error}'.format(element_id=new_entry['.id'], new_index=new_index, new_id=old_entry['.id'], error=to_native(e)))
        if modify_list or create_list or reorder_list:
            new_data = remove_dynamic(get_api_data(api_path, path_info))
    for entry in old_data:
        remove_irrelevant_data(entry, path_info)
    for entry in new_data:
        remove_irrelevant_data(entry, path_info)
    more = {}
    if module._diff:
        more['diff'] = {'before': {'data': old_data}, 'after': {'data': new_data}}
    module.exit_json(changed=bool(create_list or modify_list or remove_list or reorder_list), old_data=old_data, new_data=new_data, **more)