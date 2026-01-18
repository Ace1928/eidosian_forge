import base64
import warnings
from libcloud.utils.py3 import b, urlparse
from libcloud.compute.base import (
from libcloud.compute.types import (
from libcloud.utils.networking import is_private_subnet
from libcloud.common.cloudstack import CloudStackDriverMixIn
from libcloud.compute.providers import Provider
def ex_delete_affinity_group(self, affinity_group):
    """
        Delete an Affinity Group

        :param affinity_group: Instance of affinity group
        :type  affinity_group: :class:`CloudStackAffinityGroup`

        :rtype ``bool``
        """
    return self._async_request(command='deleteAffinityGroup', params={'id': affinity_group.id}, method='GET')['success']