from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.fusion.plugins.module_utils.fusion import (
from ansible_collections.purestorage.fusion.plugins.module_utils.startup import (
import time
import http
@_api_permission_denied_handler('storage_endpoints')
def generate_se_dict(module, fusion):
    se_dict = {}
    se_api_instance = purefusion.StorageEndpointsApi(fusion)
    az_api_instance = purefusion.AvailabilityZonesApi(fusion)
    regions_api_instance = purefusion.RegionsApi(fusion)
    regions = regions_api_instance.list_regions()
    for region in regions.items:
        azs = az_api_instance.list_availability_zones(region_name=region.name)
        for az in azs.items:
            endpoints = se_api_instance.list_storage_endpoints(region_name=region.name, availability_zone_name=az.name)
            for endpoint in endpoints.items:
                name = region.name + '/' + az.name + '/' + endpoint.name
                se_dict[name] = {'display_name': endpoint.display_name, 'endpoint_type': endpoint.endpoint_type, 'iscsi_interfaces': []}
                for iface in endpoint.iscsi.discovery_interfaces:
                    dct = {'address': iface.address, 'gateway': iface.gateway, 'mtu': iface.mtu, 'network_interface_groups': None}
                    if iface.network_interface_groups is not None:
                        dct['network_interface_groups'] = [nig.name for nig in iface.network_interface_groups]
                    se_dict[name]['iscsi_interfaces'].append(dct)
    return se_dict