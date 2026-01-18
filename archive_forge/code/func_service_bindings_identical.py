from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.netscaler.netscaler import (
import copy
def service_bindings_identical(client, module):
    log('service_bindings_identical')
    configured_service_bindings = get_configured_service_bindings(client, module)
    service_bindings = get_actual_service_bindings(client, module)
    configured_keyset = set(configured_service_bindings.keys())
    service_keyset = set(service_bindings.keys())
    if len(configured_keyset ^ service_keyset) > 0:
        return False
    for key in configured_service_bindings.keys():
        conf = configured_service_bindings[key]
        serv = service_bindings[key]
        log('s diff %s' % conf.diff_object(serv))
        if not conf.has_equal_attributes(serv):
            return False
    return True