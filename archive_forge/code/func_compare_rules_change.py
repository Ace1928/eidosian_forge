from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
from ansible.module_utils._text import to_native
def compare_rules_change(old_list, new_list, purge_list):
    old_list = old_list or []
    new_list = new_list or []
    changed = False
    for old_rule in old_list:
        matched = next((x for x in new_list if x['name'].lower() == old_rule['name'].lower()), [])
        if matched:
            changed = changed or compare_rules(old_rule, matched)
        elif not purge_list:
            new_list.append(old_rule)
        else:
            changed = True
    if not changed:
        new_names = [to_native(x['name'].lower()) for x in new_list]
        old_names = [to_native(x['name'].lower()) for x in old_list]
        changed = set(new_names) != set(old_names)
    return (changed, new_list)