from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.fusion.plugins.module_utils.fusion import (
from ansible_collections.purestorage.fusion.plugins.module_utils.getters import (
from ansible_collections.purestorage.fusion.plugins.module_utils.networking import (
from ansible_collections.purestorage.fusion.plugins.module_utils.startup import (
from ansible_collections.purestorage.fusion.plugins.module_utils.operations import (
def get_ni(module, fusion):
    """Get Network Interface or None"""
    ni_api_instance = purefusion.NetworkInterfacesApi(fusion)
    try:
        return ni_api_instance.get_network_interface(region_name=module.params['region'], availability_zone_name=module.params['availability_zone'], array_name=module.params['array'], net_intf_name=module.params['name'])
    except purefusion.rest.ApiException:
        return None