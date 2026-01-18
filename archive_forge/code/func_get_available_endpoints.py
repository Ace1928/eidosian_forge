import logging
import re
from boto.vendored.regions.exceptions import NoRegionError
def get_available_endpoints(self, service_name, partition_name='aws', allow_non_regional=False):
    result = []
    for partition in self._endpoint_data['partitions']:
        if partition['partition'] != partition_name:
            continue
        services = partition['services']
        if service_name not in services:
            continue
        for endpoint_name in services[service_name]['endpoints']:
            if allow_non_regional or endpoint_name in partition['regions']:
                result.append(endpoint_name)
    return result