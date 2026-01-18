from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.fusion.plugins.module_utils.fusion import (
from ansible_collections.purestorage.fusion.plugins.module_utils.startup import (
import time
import http
@_api_permission_denied_handler('arrays')
def generate_array_dict(module, fusion):
    array_info = {}
    array_api_instance = purefusion.ArraysApi(fusion)
    az_api_instance = purefusion.AvailabilityZonesApi(fusion)
    regions_api_instance = purefusion.RegionsApi(fusion)
    regions = regions_api_instance.list_regions()
    for region in regions.items:
        azs = az_api_instance.list_availability_zones(region_name=region.name)
        for az in azs.items:
            arrays = array_api_instance.list_arrays(availability_zone_name=az.name, region_name=region.name)
            for array in arrays.items:
                array_name = array.name
                array_space = array_api_instance.get_array_space(availability_zone_name=az.name, array_name=array_name, region_name=region.name)
                array_perf = array_api_instance.get_array_performance(availability_zone_name=az.name, array_name=array_name, region_name=region.name)
                array_info[array_name] = {'region': region.name, 'availability_zone': az.name, 'host_name': array.host_name, 'maintenance_mode': array.maintenance_mode, 'unavailable_mode': array.unavailable_mode, 'display_name': array.display_name, 'hardware_type': array.hardware_type.name, 'appliance_id': array.appliance_id, 'apartment_id': getattr(array, 'apartment_id', None), 'space': {'total_physical_space': array_space.total_physical_space}, 'performance': {'read_bandwidth': array_perf.read_bandwidth, 'read_latency_us': array_perf.read_latency_us, 'reads_per_sec': array_perf.reads_per_sec, 'write_bandwidth': array_perf.write_bandwidth, 'write_latency_us': array_perf.write_latency_us, 'writes_per_sec': array_perf.writes_per_sec}}
    return array_info