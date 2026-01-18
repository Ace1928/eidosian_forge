from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.netscaler.netscaler import (
import copy
def get_actual_service_bindings(client, module):
    log('Getting actual service bindings')
    bindings = {}
    try:
        if lbvserver_service_binding.count(client, module.params['name']) == 0:
            return bindings
    except nitro_exception as e:
        if e.errorcode == 258:
            return bindings
        else:
            raise
    bindigs_list = lbvserver_service_binding.get(client, module.params['name'])
    for item in bindigs_list:
        key = item.servicename
        bindings[key] = item
    return bindings