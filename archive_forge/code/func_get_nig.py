from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.fusion.plugins.module_utils.fusion import (
from ansible_collections.purestorage.fusion.plugins.module_utils.networking import (
from ansible_collections.purestorage.fusion.plugins.module_utils.startup import (
from ansible_collections.purestorage.fusion.plugins.module_utils.operations import (
def get_nig(module, fusion):
    """Check Network Interface Group"""
    nig_api_instance = purefusion.NetworkInterfaceGroupsApi(fusion)
    try:
        return nig_api_instance.get_network_interface_group(availability_zone_name=module.params['availability_zone'], region_name=module.params['region'], network_interface_group_name=module.params['name'])
    except purefusion.rest.ApiException:
        return None