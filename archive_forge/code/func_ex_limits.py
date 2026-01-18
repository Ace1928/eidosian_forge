import base64
import warnings
from libcloud.utils.py3 import b, urlparse
from libcloud.compute.base import (
from libcloud.compute.types import (
from libcloud.utils.networking import is_private_subnet
from libcloud.common.cloudstack import CloudStackDriverMixIn
from libcloud.compute.providers import Provider
def ex_limits(self):
    """
        Extra call to get account's resource limits, such as
        the amount of instances, volumes, snapshots and networks.

        CloudStack uses integers as the resource type so we will convert
        them to a more human readable string using the resource map

        A list of the resource type mappings can be found at
        http://goo.gl/17C6Gk

        :return: dict
        :rtype: ``dict``
        """
    result = self._sync_request(command='listResourceLimits', method='GET')
    limits = {}
    resource_map = {0: 'max_instances', 1: 'max_public_ips', 2: 'max_volumes', 3: 'max_snapshots', 4: 'max_images', 5: 'max_projects', 6: 'max_networks', 7: 'max_vpc', 8: 'max_cpu', 9: 'max_memory', 10: 'max_primary_storage', 11: 'max_secondary_storage'}
    for limit in result.get('resourcelimit', []):
        resource = resource_map.get(int(limit['resourcetype']), None)
        if not resource:
            continue
        limits[resource] = int(limit['max'])
    return limits