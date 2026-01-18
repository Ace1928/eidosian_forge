from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.netscaler.netscaler import (
import copy
def get_configured_service_bindings(client, module):
    log('Getting configured service bindings')
    readwrite_attrs = ['weight', 'name', 'servicename', 'servicegroupname']
    readonly_attrs = ['preferredlocation', 'vserverid', 'vsvrbindsvcip', 'servicetype', 'cookieipport', 'port', 'vsvrbindsvcport', 'curstate', 'ipv46', 'dynamicweight']
    configured_bindings = {}
    if 'servicebindings' in module.params and module.params['servicebindings'] is not None:
        for binding in module.params['servicebindings']:
            attribute_values_dict = copy.deepcopy(binding)
            attribute_values_dict['name'] = module.params['name']
            key = binding['servicename'].strip()
            configured_bindings[key] = ConfigProxy(actual=lbvserver_service_binding(), client=client, attribute_values_dict=attribute_values_dict, readwrite_attrs=readwrite_attrs, readonly_attrs=readonly_attrs)
    return configured_bindings