import base64
import warnings
from libcloud.utils.py3 import b, urlparse
from libcloud.compute.base import (
from libcloud.compute.types import (
from libcloud.utils.networking import is_private_subnet
from libcloud.common.cloudstack import CloudStackDriverMixIn
from libcloud.compute.providers import Provider
def ex_list_affinity_group_types(self):
    """
        List Affinity Group Types

        :rtype ``list`` of :class:`CloudStackAffinityGroupTypes`
        """
    result = self._sync_request(command='listAffinityGroupTypes', method='GET')
    if not result.get('count'):
        return []
    affinity_group_types = []
    for agt in result['affinityGroupType']:
        affinity_group_types.append(CloudStackAffinityGroupType(agt['type']))
    return affinity_group_types