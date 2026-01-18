from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.netscaler.netscaler import (
import copy
def get_actual_servicegroup_bindings(client, module):
    log('Getting actual service group bindings')
    bindings = {}
    try:
        if lbvserver_servicegroup_binding.count(client, module.params['name']) == 0:
            return bindings
    except nitro_exception as e:
        if e.errorcode == 258:
            return bindings
        else:
            raise
    bindigs_list = lbvserver_servicegroup_binding.get(client, module.params['name'])
    for item in bindigs_list:
        key = item.servicegroupname
        bindings[key] = item
    return bindings