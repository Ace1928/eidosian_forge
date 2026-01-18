from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.netscaler.netscaler import (
def get_configured_policybindings(client, module):
    log('Getting configured policy bindigs')
    bindings = {}
    if module.params['policybindings'] is None:
        return bindings
    for binding in module.params['policybindings']:
        binding['name'] = module.params['name']
        key = binding['policyname']
        binding_proxy = ConfigProxy(actual=csvserver_cspolicy_binding(), client=client, readwrite_attrs=['priority', 'bindpoint', 'policyname', 'labelname', 'gotopriorityexpression', 'targetlbvserver', 'name', 'invoke', 'labeltype'], readonly_attrs=[], attribute_values_dict=binding)
        bindings[key] = binding_proxy
    return bindings