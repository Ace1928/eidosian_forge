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
def add_instances(self, node_list):
    """
        Adds a list of instances to the specified instance group. All of the
        instances in the instance group must be in the same
        network/subnetwork. Read  Adding instances for more information.

        Scopes needed - one of the following:
        * https://www.googleapis.com/auth/cloud-platform
        * https://www.googleapis.com/auth/compute

        :param  instancegroup:  The Instance Group where you are
                                adding instances.
        :type   instancegroup: :class:``GCEInstanceGroup``

        :param  node_list: List of nodes to add.
        :type   node_list: ``list`` of :class:`Node` or ``list`` of
                           :class:`GCENode`

        :return:  Return True if successful.
        :rtype: ``bool``
        """
    return self.driver.ex_instancegroup_add_instances(instancegroup=self, node_list=node_list)