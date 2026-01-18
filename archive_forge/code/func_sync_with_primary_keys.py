from __future__ import (absolute_import, division, print_function)
from collections import defaultdict
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible.module_utils.common.text.converters import to_native
from ansible_collections.community.routeros.plugins.module_utils.api import (
from ansible_collections.community.routeros.plugins.module_utils._api_data import (
def sync_with_primary_keys(module, api, path, path_info):
    primary_keys = path_info.primary_keys
    if path_info.fixed_entries:
        if module.params['ensure_order']:
            module.fail_json(msg='ensure_order=true cannot be used with this path')
        if module.params['handle_absent_entries'] == 'remove':
            module.fail_json(msg='handle_absent_entries=remove cannot be used with this path')
    data = module.params['data']
    new_data_by_key = OrderedDict()
    for index, entry in enumerate(data):
        for primary_key in primary_keys:
            if primary_key not in entry:
                module.fail_json(msg='Every element in data must contain "{primary_key}". For example, the element at index #{index} does not provide it.'.format(primary_key=primary_key, index=index + 1))
        pks = tuple((entry[primary_key] for primary_key in primary_keys))
        if pks in new_data_by_key:
            module.fail_json(msg='Every element in data must contain a unique value for {primary_keys}. The value {value} appears at least twice.'.format(primary_keys=','.join(primary_keys), value=','.join(['"{0}"'.format(pk) for pk in pks])))
        polish_entry(entry, path_info, module, ' for {values}'.format(values=', '.join(['{primary_key}="{value}"'.format(primary_key=primary_key, value=value) for primary_key, value in zip(primary_keys, pks)])))
        new_data_by_key[pks] = entry
    api_path = compose_api_path(api, path)
    old_data = get_api_data(api_path, path_info)
    old_data = remove_dynamic(old_data)
    old_data_by_key = OrderedDict()
    id_by_key = {}
    for entry in old_data:
        pks = tuple((entry[primary_key] for primary_key in primary_keys))
        old_data_by_key[pks] = entry
        id_by_key[pks] = entry['.id']
    new_data = []
    create_list = []
    modify_list = []
    remove_list = []
    remove_keys = []
    handle_absent_entries = module.params['handle_absent_entries']
    for key, old_entry in old_data_by_key.items():
        new_entry = new_data_by_key.pop(key, None)
        if new_entry is None:
            if handle_absent_entries == 'remove':
                remove_list.append(old_entry['.id'])
                remove_keys.append(key)
            else:
                new_data.append(old_entry)
        else:
            modifications, updated_entry = find_modifications(old_entry, new_entry, path_info, module, ' for {values}'.format(values=', '.join(['{primary_key}="{value}"'.format(primary_key=primary_key, value=value) for primary_key, value in zip(primary_keys, key)])))
            new_data.append(updated_entry)
            if modifications:
                modifications['.id'] = old_entry['.id']
                modify_list.append((key, modifications))
    for new_entry in new_data_by_key.values():
        if path_info.fixed_entries:
            module.fail_json(msg='Cannot add new entry {values} to this path'.format(values=', '.join(['{primary_key}="{value}"'.format(primary_key=primary_key, value=new_entry[primary_key]) for primary_key in primary_keys])))
        remove_read_only(new_entry, path_info)
        create_list.append(new_entry)
        new_entry = new_entry.copy()
        for key in list(new_entry):
            if key.startswith('!'):
                new_entry.pop(key)
        new_data.append(new_entry)
    reorder_list = []
    if module.params['ensure_order']:
        index_by_key = dict()
        for index, entry in enumerate(new_data):
            index_by_key[tuple((entry[primary_key] for primary_key in primary_keys))] = index
        for index, source_entry in enumerate(data):
            source_pks = tuple((source_entry[primary_key] for primary_key in primary_keys))
            source_index = index_by_key.pop(source_pks)
            if index == source_index:
                continue
            entry = new_data[index]
            pks = tuple((entry[primary_key] for primary_key in primary_keys))
            reorder_list.append((source_pks, index, pks))
            for k, v in index_by_key.items():
                if v >= index and v < source_index:
                    index_by_key[k] = v + 1
            new_data.insert(index, new_data.pop(source_index))
    if not module.check_mode:
        if remove_list:
            try:
                api_path.remove(*remove_list)
            except (LibRouterosError, UnicodeEncodeError) as e:
                module.fail_json(msg='Error while removing {remove_list}: {error}'.format(remove_list=', '.join(['{identifier} (ID {id})'.format(identifier=format_pk(primary_keys, key), id=id) for id, key in zip(remove_list, remove_keys)]), error=to_native(e)))
        for key, modifications in modify_list:
            try:
                api_path.update(**modifications)
            except (LibRouterosError, UnicodeEncodeError) as e:
                module.fail_json(msg='Error while modifying for {identifier} (ID {id}): {error}'.format(identifier=format_pk(primary_keys, key), id=modifications['.id'], error=to_native(e)))
        for entry in create_list:
            try:
                entry['.id'] = api_path.add(**prepare_for_add(entry, path_info))
                pks = tuple((entry[primary_key] for primary_key in primary_keys))
                id_by_key[pks] = entry['.id']
            except (LibRouterosError, UnicodeEncodeError) as e:
                module.fail_json(msg='Error while creating entry for {identifier}: {error}'.format(identifier=format_pk(primary_keys, [entry[pk] for pk in primary_keys]), error=to_native(e)))
        for element_pks, new_index, new_pks in reorder_list:
            try:
                element_id = id_by_key[element_pks]
                new_id = id_by_key[new_pks]
                for res in api_path('move', numbers=element_id, destination=new_id):
                    pass
            except (LibRouterosError, UnicodeEncodeError) as e:
                module.fail_json(msg='Error while moving entry ID {element_id} to position of ID {new_id}: {error}'.format(element_id=element_id, new_id=new_id, error=to_native(e)))
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