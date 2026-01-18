from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
import copy
from ansible_collections.community.network.plugins.module_utils.network.netscaler.netscaler import (ConfigProxy, get_nitro_client, netscaler_common_arguments,
def servicegroup_identical(client, module, servicegroup_proxy):
    log('Checking if service group is identical')
    servicegroups = servicegroup.get_filtered(client, 'servicegroupname:%s' % module.params['servicegroupname'])
    if servicegroup_proxy.has_equal_attributes(servicegroups[0]):
        return True
    else:
        return False