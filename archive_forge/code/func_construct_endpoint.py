import logging
import re
from boto.vendored.regions.exceptions import NoRegionError
def construct_endpoint(self, service_name, region_name=None):
    for partition in self._endpoint_data['partitions']:
        result = self._endpoint_for_partition(partition, service_name, region_name)
        if result:
            return result