from __future__ import absolute_import, division, print_function
import copy
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.netscaler.netscaler import (
def get_configured_domain_bindings_proxys(client, module):
    log('get_configured_domain_bindings_proxys')
    configured_domain_proxys = {}
    if module.params['domain_bindings'] is not None:
        for configured_domain_binding in module.params['domain_bindings']:
            binding_values = copy.deepcopy(configured_domain_binding)
            binding_values['name'] = module.params['name']
            gslbvserver_domain_binding_proxy = ConfigProxy(actual=gslbvserver_domain_binding(), client=client, attribute_values_dict=binding_values, readwrite_attrs=gslbvserver_domain_binding_rw_attrs, readonly_attrs=[])
            configured_domain_proxys[configured_domain_binding['domainname']] = gslbvserver_domain_binding_proxy
    return configured_domain_proxys