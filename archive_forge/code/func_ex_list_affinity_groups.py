import base64
import warnings
from libcloud.utils.py3 import b, urlparse
from libcloud.compute.base import (
from libcloud.compute.types import (
from libcloud.utils.networking import is_private_subnet
from libcloud.common.cloudstack import CloudStackDriverMixIn
from libcloud.compute.providers import Provider
def ex_list_affinity_groups(self):
    """
        List Affinity Groups

        :rtype ``list`` of :class:`CloudStackAffinityGroup`
        """
    result = self._sync_request(command='listAffinityGroups', method='GET')
    if not result.get('count'):
        return []
    affinity_groups = []
    for ag in result['affinitygroup']:
        affinity_groups.append(self._to_affinity_group(ag))
    return affinity_groups