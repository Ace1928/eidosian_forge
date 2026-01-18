from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
import copy
from ansible_collections.community.network.plugins.module_utils.network.netscaler.netscaler import (ConfigProxy, get_nitro_client, netscaler_common_arguments,
def get_actual_service_members(client, module):
    try:
        count = servicegroup_servicegroupmember_binding.count(client, module.params['servicegroupname'])
        if count > 0:
            servicegroup_members = servicegroup_servicegroupmember_binding.get(client, module.params['servicegroupname'])
        else:
            servicegroup_members = []
    except nitro_exception as e:
        if e.errorcode == 258:
            servicegroup_members = []
        else:
            raise
    return servicegroup_members