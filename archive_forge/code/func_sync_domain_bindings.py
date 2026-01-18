from __future__ import absolute_import, division, print_function
import copy
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.netscaler.netscaler import (
def sync_domain_bindings(client, module):
    log('sync_domain_bindings')
    actual_domain_bindings = get_actual_domain_bindings(client, module)
    configured_domain_proxys = get_configured_domain_bindings_proxys(client, module)
    for domainname, actual_domain_binding in actual_domain_bindings.items():
        if domainname not in configured_domain_proxys.keys():
            log('Deleting absent binding for domain %s' % domainname)
            gslbvserver_domain_binding.delete(client, actual_domain_binding)
    for proxy_key, binding_proxy in configured_domain_proxys.items():
        if proxy_key in actual_domain_bindings:
            actual_binding = actual_domain_bindings[proxy_key]
            if not binding_proxy.has_equal_attributes(actual_binding):
                log('Deleting differing binding for domain %s' % binding_proxy.domainname)
                gslbvserver_domain_binding.delete(client, actual_binding)
                log('Adding anew binding for domain %s' % binding_proxy.domainname)
                binding_proxy.add()
    for proxy_key, binding_proxy in configured_domain_proxys.items():
        if proxy_key not in actual_domain_bindings.keys():
            log('Adding domain binding for domain %s' % binding_proxy.domainname)
            binding_proxy.add()