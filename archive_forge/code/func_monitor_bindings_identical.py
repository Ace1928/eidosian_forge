from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
import copy
from ansible_collections.community.network.plugins.module_utils.network.netscaler.netscaler import (ConfigProxy, get_nitro_client, netscaler_common_arguments,
def monitor_bindings_identical(client, module):
    log('Entering monitor_bindings_identical')
    configured_bindings = get_configured_monitor_bindings(client, module)
    actual_bindings = get_actual_monitor_bindings(client, module)
    configured_key_set = set(configured_bindings.keys())
    actual_key_set = set(actual_bindings.keys())
    symmetrical_diff = configured_key_set ^ actual_key_set
    for default_monitor in ('tcp-default', 'ping-default'):
        if default_monitor in symmetrical_diff:
            log('Excluding %s monitor from key comparison' % default_monitor)
            symmetrical_diff.remove(default_monitor)
    if len(symmetrical_diff) > 0:
        return False
    for key in configured_key_set:
        configured_proxy = configured_bindings[key]
        if not hasattr(configured_proxy, 'weight'):
            configured_proxy.weight = 1
        log('configured_proxy %s' % [configured_proxy.monitorname, configured_proxy.servicegroupname, configured_proxy.weight])
        log('actual_bindings %s' % [actual_bindings[key].monitor_name, actual_bindings[key].servicegroupname, actual_bindings[key].weight])
        if not monitor_binding_equal(configured_proxy, actual_bindings[key]):
            return False
    return True