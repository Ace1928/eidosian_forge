from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
import copy
from ansible_collections.community.network.plugins.module_utils.network.netscaler.netscaler import (ConfigProxy, get_nitro_client, netscaler_common_arguments,
def servicemembers_identical(client, module):
    log('servicemembers_identical')
    servicegroup_members = get_actual_service_members(client, module)
    log('servicemembers %s' % servicegroup_members)
    module_servicegroups = get_configured_service_members(client, module)
    log('Number of service group members %s' % len(servicegroup_members))
    if len(servicegroup_members) != len(module_servicegroups):
        return False
    identical_count = 0
    for actual_member in servicegroup_members:
        for member in module_servicegroups:
            if member.has_equal_attributes(actual_member):
                identical_count += 1
                break
    if identical_count != len(servicegroup_members):
        return False
    return True