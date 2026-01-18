from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.netscaler.netscaler import (
def get_actual_policybindings(client, module):
    log('Getting actual policy bindigs')
    bindings = {}
    try:
        count = csvserver_cspolicy_binding.count(client, name=module.params['name'])
        if count == 0:
            return bindings
    except nitro_exception as e:
        if e.errorcode == 258:
            return bindings
        else:
            raise
    for binding in csvserver_cspolicy_binding.get(client, name=module.params['name']):
        key = binding.policyname
        bindings[key] = binding
    return bindings