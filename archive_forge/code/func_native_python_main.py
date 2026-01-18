from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.api import basic_auth_argument_spec
from ansible_collections.community.general.plugins.module_utils.gitlab import (
def native_python_main(this_gitlab, purge, requested_variables, state, module):
    change = False
    return_value = dict(added=[], updated=[], removed=[], untouched=[])
    gitlab_keys = this_gitlab.list_all_project_variables()
    before = [x.attributes for x in gitlab_keys]
    gitlab_keys = this_gitlab.list_all_project_variables()
    existing_variables = filter_returned_variables(gitlab_keys)
    for item in requested_variables:
        item['key'] = item.pop('name')
        item['value'] = str(item.get('value'))
        if item.get('protected') is None:
            item['protected'] = False
        if item.get('raw') is None:
            item['raw'] = False
        if item.get('masked') is None:
            item['masked'] = False
        if item.get('environment_scope') is None:
            item['environment_scope'] = '*'
        if item.get('variable_type') is None:
            item['variable_type'] = 'env_var'
    if module.check_mode:
        untouched, updated, added = compare(requested_variables, existing_variables, state)
    if state == 'present':
        add_or_update = [x for x in requested_variables if x not in existing_variables]
        for item in add_or_update:
            try:
                if this_gitlab.create_variable(item):
                    return_value['added'].append(item)
            except Exception:
                if this_gitlab.update_variable(item):
                    return_value['updated'].append(item)
        if purge:
            gitlab_keys = this_gitlab.list_all_project_variables()
            existing_variables = filter_returned_variables(gitlab_keys)
            remove = [x for x in existing_variables if x not in requested_variables]
            for item in remove:
                if this_gitlab.delete_variable(item):
                    return_value['removed'].append(item)
    elif state == 'absent':
        for item in existing_variables:
            item.pop('value')
            item.pop('variable_type')
        for item in requested_variables:
            item.pop('value')
            item.pop('variable_type')
        if not purge:
            remove_requested = [x for x in requested_variables if x in existing_variables]
            for item in remove_requested:
                if this_gitlab.delete_variable(item):
                    return_value['removed'].append(item)
        else:
            for item in existing_variables:
                if this_gitlab.delete_variable(item):
                    return_value['removed'].append(item)
    if module.check_mode:
        return_value = dict(added=added, updated=updated, removed=return_value['removed'], untouched=untouched)
    if any((return_value[x] for x in ['added', 'removed', 'updated'])):
        change = True
    gitlab_keys = this_gitlab.list_all_project_variables()
    after = [x.attributes for x in gitlab_keys]
    return (change, return_value, before, after)