from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.fusion.plugins.module_utils.fusion import (
from ansible_collections.purestorage.fusion.plugins.module_utils.startup import (
import time
import http
@_api_permission_denied_handler('network_interface_groups')
def generate_nigs_dict(module, fusion):
    nigs_dict = {}
    nig_api_instance = purefusion.NetworkInterfaceGroupsApi(fusion)
    az_api_instance = purefusion.AvailabilityZonesApi(fusion)
    regions_api_instance = purefusion.RegionsApi(fusion)
    regions = regions_api_instance.list_regions()
    for region in regions.items:
        azs = az_api_instance.list_availability_zones(region_name=region.name)
        for az in azs.items:
            nigs = nig_api_instance.list_network_interface_groups(region_name=region.name, availability_zone_name=az.name)
            for nig in nigs.items:
                name = region.name + '/' + az.name + '/' + nig.name
                nigs_dict[name] = {'display_name': nig.display_name, 'gateway': nig.eth.gateway, 'prefix': nig.eth.prefix, 'mtu': nig.eth.mtu}
    return nigs_dict