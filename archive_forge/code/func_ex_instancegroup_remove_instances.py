import sys
import time
import datetime
import itertools
from libcloud.pricing import get_pricing
from libcloud.common.base import LazyObject
from libcloud.common.types import LibcloudError
from libcloud.compute.base import (
from libcloud.common.google import (
from libcloud.compute.types import NodeState
from libcloud.utils.iso8601 import parse_date
from libcloud.compute.providers import Provider
def ex_instancegroup_remove_instances(self, instancegroup, node_list):
    """
        Removes one or more instances from the specified instance group,
        but does not delete those instances.

        Scopes needed - one of the following:
        * https://www.googleapis.com/auth/cloud-platform
        * https://www.googleapis.com/auth/compute

        :param  instancegroup:  The Instance Group where the
                                specified instances will be removed.
        :type   instancegroup: :class:``GCEInstanceGroup``

        :param  node_list: List of nodes to add.
        :type   node_list: ``list`` of :class:`Node` or ``list`` of
                           :class:`GCENode`

        :return:  True if successful.
        :rtype: ``bool``
        """
    request = '/zones/{}/instanceGroups/{}/removeInstances'.format(instancegroup.zone.name, instancegroup.name)
    request_data = {'instances': [{'instance': x.extra['selfLink']} for x in node_list]}
    self.connection.async_request(request, method='POST', data=request_data)
    return True