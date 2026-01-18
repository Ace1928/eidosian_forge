from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
from ansible.module_utils._text import to_native
def compare_rules(old_rule, rule):

    def compare_list_rule(old_rule, rule, key):
        return set(map(str, rule.get(key) or [])) != set(map(str, old_rule.get(key) or []))
    changed = False
    if old_rule['name'].lower() != rule['name'].lower():
        changed = True
    if rule.get('description', None) != old_rule['description']:
        changed = True
    if rule['protocol'].lower() != old_rule['protocol'].lower():
        changed = True
    if str(rule['source_port_range']) != str(old_rule['source_port_range']):
        changed = True
    if str(rule['destination_port_range']) != str(old_rule['destination_port_range']):
        changed = True
    if rule['access'] != old_rule['access']:
        changed = True
    if rule['priority'] != old_rule['priority']:
        changed = True
    if rule['direction'] != old_rule['direction']:
        changed = True
    if str(rule['source_address_prefix']) != str(old_rule['source_address_prefix']):
        changed = True
    if str(rule['destination_address_prefix']) != str(old_rule['destination_address_prefix']):
        changed = True
    if compare_list_rule(old_rule, rule, 'source_address_prefixes'):
        changed = True
    if compare_list_rule(old_rule, rule, 'destination_address_prefixes'):
        changed = True
    if compare_list_rule(old_rule, rule, 'source_port_ranges'):
        changed = True
    if compare_list_rule(old_rule, rule, 'destination_port_ranges'):
        changed = True
    if compare_list_rule(old_rule, rule, 'source_application_security_groups'):
        changed = True
    if compare_list_rule(old_rule, rule, 'destination_application_security_groups'):
        changed = True
    return changed