from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.netscaler.netscaler import (
import copy
def get_configured_servicegroup_bindings(client, module):
    log('Getting configured service group bindings')
    readwrite_attrs = ['weight', 'name', 'servicename', 'servicegroupname']
    readonly_attrs = []
    configured_bindings = {}
    if 'servicegroupbindings' in module.params and module.params['servicegroupbindings'] is not None:
        for binding in module.params['servicegroupbindings']:
            attribute_values_dict = copy.deepcopy(binding)
            attribute_values_dict['name'] = module.params['name']
            key = binding['servicegroupname'].strip()
            configured_bindings[key] = ConfigProxy(actual=lbvserver_servicegroup_binding(), client=client, attribute_values_dict=attribute_values_dict, readwrite_attrs=readwrite_attrs, readonly_attrs=readonly_attrs)
    return configured_bindings