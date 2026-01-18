import base64
import warnings
from libcloud.utils.py3 import b, urlparse
from libcloud.compute.base import (
from libcloud.compute.types import (
from libcloud.utils.networking import is_private_subnet
from libcloud.common.cloudstack import CloudStackDriverMixIn
from libcloud.compute.providers import Provider
def ex_update_node_affinity_group(self, node, affinity_group_list):
    """
        Updates the affinity/anti-affinity group associations of a virtual
        machine. The VM has to be stopped and restarted for the new properties
        to take effect.

        :param node: Node to update.
        :type node: :class:`CloudStackNode`

        :param affinity_group_list: List of CloudStackAffinityGroup to
                                    associate
        :type affinity_group_list: ``list`` of :class:`CloudStackAffinityGroup`

        :rtype :class:`CloudStackNode`
        """
    affinity_groups = ','.join((ag.id for ag in affinity_group_list))
    result = self._async_request(command='updateVMAffinityGroup', params={'id': node.id, 'affinitygroupids': affinity_groups}, method='GET')
    return self._to_node(data=result['virtualmachine'])