from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.netscaler.netscaler import (
def get_default_lb_vserver(client, module):
    try:
        default_lb_vserver = csvserver_lbvserver_binding.get(client, module.params['name'])
        return default_lb_vserver[0]
    except nitro_exception as e:
        if e.errorcode == 258:
            return csvserver_lbvserver_binding()
        else:
            raise